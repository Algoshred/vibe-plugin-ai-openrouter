/**
 * vibe-plugin-ai-openrouter
 *
 * OpenRouter AI agent provider for VibeControls Agent.
 * Implements the AIAgentProvider interface in SDK-only mode.
 *
 * OpenRouter ships an SDK (`@openrouter/sdk`) but no first-party CLI, so this
 * plugin exposes a single `sdk` mode. Auth is read from the
 * `OPENROUTER_API_KEY` env var or from agent configuration via host services.
 *
 * Models are fetched live from `https://openrouter.ai/api/v1/models` (cached
 * for 1 hour) and mapped into the AIModelInfo shape.
 */

import { Elysia } from "elysia";
import type { HostServices, VibePlugin } from "@vibecontrols/plugin-sdk";
import {
  BoundLogger,
  ProviderRegistry,
  TelemetryEmitter,
  createLifecycleHooks,
} from "@vibecontrols/plugin-sdk";

// ── AI Provider Contract Types ──────────────────────────────────────────
// (provider-specific contract — kept inline; not part of the SDK surface)

type ProviderMode = "sdk" | "cli";

interface AIModelInfo {
  id: string;
  name: string;
  provider: string;
  contextWindow: number;
  maxOutputTokens: number;
  supportsVision: boolean;
  supportsStreaming: boolean;
  inputPricePerMToken: number;
  outputPricePerMToken: number;
}

interface AIProviderCapabilities {
  streaming: boolean;
  vision: boolean;
  fileAttachments: boolean;
  toolUse: boolean;
  mcpSupport: boolean;
  voiceMode: boolean;
  cancelSupport: boolean;
  modelListing: boolean;
}

interface AIFileAttachment {
  filename: string;
  mimeType: string;
  content: Buffer | string;
  size: number;
}

type AISessionStatus =
  | "active"
  | "idle"
  | "processing"
  | "error"
  | "terminated";
type AILogType =
  | "input"
  | "output"
  | "thinking"
  | "event"
  | "error"
  | "metadata";

interface AISessionConfig {
  name: string;
  agentType: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
  workingDirectory?: string;
  providerConfig?: Record<string, unknown>;
}

interface AISession {
  id: string;
  name: string;
  status: AISessionStatus;
  agentType: string;
  provider: string;
  config: AISessionConfig;
  stats: AIUsageStats;
  createdAt: string;
  updatedAt: string;
}

interface AIContext {
  id: string;
  type: string;
  content: string;
  metadata?: Record<string, unknown>;
}

interface AIResponse {
  content: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  thinkingSteps?: string[];
  durationMs: number;
  metadata?: Record<string, unknown>;
}

interface AIStreamChunk {
  type: "text" | "thinking" | "error" | "done";
  content: string;
  tokensUsed?: number;
}

interface AILog {
  id: string;
  sessionId: string;
  type: AILogType;
  content: string;
  tokenCount?: number;
  model?: string;
  durationMs?: number;
  agentMetadata?: Record<string, unknown>;
  createdAt: string;
}

interface AILogFilter {
  types?: AILogType[];
  startDate?: string;
  endDate?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

interface AIUsageStats {
  inputTokens: number;
  outputTokens: number;
  requestCount: number;
  estimatedCostUsd: number;
  modelBreakdown?: Record<
    string,
    { inputTokens: number; outputTokens: number; requestCount: number }
  >;
}

interface AIAgentProvider {
  readonly name: string;
  createSession(config: AISessionConfig): Promise<AISession>;
  sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse>;
  streamPrompt?(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse>;
  getSessionLogs(sessionId: string, filter?: AILogFilter): Promise<AILog[]>;
  getUsageStats(sessionId: string): Promise<AIUsageStats>;
  configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void>;
  destroySession(sessionId: string): Promise<void>;
  listSessions(): Promise<AISession[]>;
  getSessionStatus(sessionId: string): Promise<AISessionStatus>;
  healthCheck(): Promise<{ ok: boolean; message?: string }>;
  listModels?(): Promise<AIModelInfo[]>;
  cancelRequest?(sessionId: string): Promise<void>;
  getCapabilities?(): AIProviderCapabilities;
  attachFiles?(sessionId: string, files: AIFileAttachment[]): Promise<void>;
  getMode?(): ProviderMode;
  setMode?(mode: ProviderMode): void;
  getCliLaunchSpec(): {
    binary: string;
    baseArgs?: string[];
    env?: Record<string, string>;
  } | null;
  sdkOneShot(opts: {
    prompt: string;
    model?: string;
    maxTokens?: number;
    extras?: Record<string, unknown>;
  }): Promise<{ text: string; usage?: unknown }>;
}

// Log ingester interface (from ai plugin's service registry)
interface LogIngester {
  append(input: {
    sessionId: string;
    type: AILogType;
    content: string;
    tokenCount?: number;
    model?: string;
    durationMs?: number;
    agentMetadata?: Record<string, unknown>;
  }): unknown;
}

// ── Provider Adapter Interface ──────────────────────────────────────────

interface ProviderAdapter {
  readonly mode: ProviderMode;

  sendPrompt(
    prompt: string,
    config: AISessionConfig,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }>;

  streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
    signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }>;

  healthCheck(): Promise<{ ok: boolean; message?: string }>;
}

// ── Constants ───────────────────────────────────────────────────────────

const PROVIDER_NAME = "openrouter";
const DISPLAY_NAME = "OpenRouter";
const DEFAULT_MODEL = "openai/gpt-4o-mini";
const DEFAULT_MAX_TOKENS = 4096;
const API_PREFIX = `/api/ai-${PROVIDER_NAME}`;
const SUPPORTED_MODES: ProviderMode[] = ["sdk"];
const OPENROUTER_API_BASE = "https://openrouter.ai/api/v1";
const MODELS_CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

// ── Models cache (1 hour, in-memory) ─────────────────────────────────────

interface OpenRouterApiModel {
  id: string;
  name?: string;
  context_length?: number;
  top_provider?: { max_completion_tokens?: number | null } | null;
  architecture?: { input_modalities?: string[] | null } | null;
  pricing?: { prompt?: string | null; completion?: string | null } | null;
}

interface ModelsCacheEntry {
  fetchedAt: number;
  models: AIModelInfo[];
}

let modelsCache: ModelsCacheEntry | null = null;

function mapApiModelToInfo(m: OpenRouterApiModel): AIModelInfo {
  const inputPriceUsdPerToken = parseFloat(m.pricing?.prompt ?? "0") || 0;
  const outputPriceUsdPerToken = parseFloat(m.pricing?.completion ?? "0") || 0;
  const inputModalities = m.architecture?.input_modalities ?? [];
  return {
    id: m.id,
    name: m.name || m.id,
    provider: PROVIDER_NAME,
    contextWindow: m.context_length ?? 0,
    maxOutputTokens: m.top_provider?.max_completion_tokens ?? 0,
    supportsVision: inputModalities.includes("image"),
    supportsStreaming: true,
    // OpenRouter prices are USD per token; convert to per-million.
    inputPricePerMToken: inputPriceUsdPerToken * 1_000_000,
    outputPricePerMToken: outputPriceUsdPerToken * 1_000_000,
  };
}

async function fetchOpenRouterModels(): Promise<AIModelInfo[]> {
  const now = Date.now();
  if (modelsCache && now - modelsCache.fetchedAt < MODELS_CACHE_TTL_MS) {
    return modelsCache.models;
  }
  const res = await fetch(`${OPENROUTER_API_BASE}/models`);
  if (!res.ok) {
    throw new Error(
      `Failed to fetch OpenRouter models: HTTP ${res.status} ${res.statusText}`,
    );
  }
  const json = (await res.json()) as { data?: OpenRouterApiModel[] };
  const models = (json.data ?? []).map(mapApiModelToInfo);
  modelsCache = { fetchedAt: now, models };
  return models;
}

// ── SDK Adapter ─────────────────────────────────────────────────────────

interface OpenRouterChatChoice {
  message?: { content?: string | null } | null;
}
interface OpenRouterChatResult {
  choices: OpenRouterChatChoice[];
  model: string;
  usage?: {
    promptTokens?: number | null;
    completionTokens?: number | null;
  } | null;
}
interface OpenRouterStreamChoice {
  delta?: { content?: string | null } | null;
}
interface OpenRouterStreamChunk {
  choices: OpenRouterStreamChoice[];
  model?: string;
  usage?: {
    promptTokens?: number | null;
    completionTokens?: number | null;
  } | null;
}

interface OpenRouterClient {
  chat: {
    send(request: {
      chatRequest: {
        model: string;
        messages: Array<{ role: string; content: string }>;
        maxTokens?: number;
        temperature?: number;
        stream?: boolean;
      };
    }): Promise<OpenRouterChatResult | AsyncIterable<OpenRouterStreamChunk>>;
  };
}

type OpenRouterAuthResolver = () => Promise<string | undefined>;

class OpenRouterSdkAdapter implements ProviderAdapter {
  readonly mode: ProviderMode = "sdk";
  private client: OpenRouterClient | null = null;
  private resolveAuth: OpenRouterAuthResolver;

  constructor(resolveAuth: OpenRouterAuthResolver) {
    this.resolveAuth = resolveAuth;
  }

  private async getClient(): Promise<OpenRouterClient> {
    if (this.client) return this.client;

    const apiKey = await this.resolveAuth();
    if (!apiKey) {
      throw new Error(
        "OpenRouter SDK auth is not configured. Set OPENROUTER_API_KEY, " +
          "or store it in agent config (e.g. POST /api/config { key: " +
          "'OPENROUTER_API_KEY', value: '...' }).",
      );
    }

    let mod: unknown;
    try {
      mod = await import("@openrouter/sdk");
    } catch {
      throw new Error(
        "Failed to load @openrouter/sdk. Install it with: bun add @openrouter/sdk",
      );
    }

    const m = mod as { OpenRouter?: unknown; default?: unknown };
    const Ctor = (m.OpenRouter ?? m.default) as
      | (new (opts: { apiKey: string }) => OpenRouterClient)
      | undefined;
    if (!Ctor) {
      throw new Error(
        "@openrouter/sdk did not expose an OpenRouter constructor",
      );
    }
    this.client = new Ctor({ apiKey });
    return this.client;
  }

  private buildMessages(
    prompt: string,
    config: AISessionConfig,
  ): Array<{ role: string; content: string }> {
    const messages: Array<{ role: string; content: string }> = [];
    if (config.systemPrompt) {
      messages.push({ role: "system", content: config.systemPrompt });
    }
    messages.push({ role: "user", content: prompt });
    return messages;
  }

  async sendPrompt(
    prompt: string,
    config: AISessionConfig,
    _signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const client = await this.getClient();
    const startTime = Date.now();
    const model = config.model || DEFAULT_MODEL;

    const result = (await client.chat.send({
      chatRequest: {
        model,
        messages: this.buildMessages(prompt, config),
        maxTokens: config.maxTokens ?? DEFAULT_MAX_TOKENS,
        ...(config.temperature !== undefined
          ? { temperature: config.temperature }
          : {}),
        stream: false,
      },
    })) as OpenRouterChatResult;

    const durationMs = Date.now() - startTime;
    const content = result.choices
      .map((c) => c.message?.content ?? "")
      .join("");

    return {
      content,
      model: result.model,
      inputTokens: result.usage?.promptTokens ?? 0,
      outputTokens: result.usage?.completionTokens ?? 0,
      durationMs,
      metadata: { provider: PROVIDER_NAME, mode: "sdk" },
    };
  }

  async streamPrompt(
    prompt: string,
    config: AISessionConfig,
    onChunk: (chunk: AIStreamChunk) => void,
    _signal?: AbortSignal,
  ): Promise<{
    content: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    durationMs: number;
    metadata?: Record<string, unknown>;
  }> {
    const client = await this.getClient();
    const startTime = Date.now();
    const model = config.model || DEFAULT_MODEL;

    const stream = (await client.chat.send({
      chatRequest: {
        model,
        messages: this.buildMessages(prompt, config),
        maxTokens: config.maxTokens ?? DEFAULT_MAX_TOKENS,
        ...(config.temperature !== undefined
          ? { temperature: config.temperature }
          : {}),
        stream: true,
      },
    })) as AsyncIterable<OpenRouterStreamChunk>;

    let fullContent = "";
    let resolvedModel = model;
    let inputTokens = 0;
    let outputTokens = 0;

    try {
      for await (const chunk of stream) {
        if (chunk.model) resolvedModel = chunk.model;
        if (chunk.usage) {
          inputTokens = chunk.usage.promptTokens ?? inputTokens;
          outputTokens = chunk.usage.completionTokens ?? outputTokens;
        }
        for (const choice of chunk.choices ?? []) {
          const piece = choice.delta?.content ?? "";
          if (piece) {
            fullContent += piece;
            onChunk({ type: "text", content: piece });
          }
        }
      }
    } catch (err) {
      onChunk({
        type: "error",
        content: err instanceof Error ? err.message : "stream error",
      });
      throw err;
    }

    onChunk({ type: "done", content: "" });
    const durationMs = Date.now() - startTime;

    return {
      content: fullContent,
      model: resolvedModel,
      inputTokens,
      outputTokens,
      durationMs,
      metadata: { provider: PROVIDER_NAME, mode: "sdk" },
    };
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    try {
      await this.getClient();
      return {
        ok: true,
        message: `${DISPLAY_NAME} SDK ready (API key configured)`,
      };
    } catch (err) {
      return {
        ok: false,
        message:
          err instanceof Error ? err.message : "SDK initialization failed",
      };
    }
  }
}

// ── Provider Implementation ─────────────────────────────────────────────

interface ManagedSession {
  id: string;
  config: AISessionConfig;
  status: AISessionStatus;
  stats: AIUsageStats;
  abortController: AbortController | null;
  files: AIFileAttachment[];
  createdAt: string;
  updatedAt: string;
}

class OpenRouterProvider implements AIAgentProvider {
  readonly name = PROVIDER_NAME;
  private sessions = new Map<string, ManagedSession>();
  private logIngester: LogIngester | null = null;
  private hostServices: HostServices | null = null;
  private logger: BoundLogger | null = null;
  private adapter: ProviderAdapter | null = null;
  private cachedApiKey: string | undefined;

  setHostServices(hs: HostServices): void {
    this.hostServices = hs;
    this.logger = new BoundLogger(hs.logger, `${PROVIDER_NAME}-provider`);
    const registry = new ProviderRegistry(hs);
    this.logIngester =
      registry.getProvider<LogIngester>("ai", "log-ingester") ?? null;

    void Promise.resolve(hs.getConfig?.("OPENROUTER_API_KEY"))
      .then((apiKey) => {
        const trimmed = apiKey?.trim();
        if (trimmed) this.cachedApiKey = trimmed;
      })
      .catch(() => {});
  }

  getSupportedModes(): ProviderMode[] {
    return [...SUPPORTED_MODES];
  }

  getDisplayName(): string {
    return DISPLAY_NAME;
  }

  getPrereqApiPrefix(): string {
    return API_PREFIX;
  }

  private async resolveAuth(): Promise<string | undefined> {
    const envApiKey = process.env["OPENROUTER_API_KEY"]?.trim();
    if (envApiKey) return envApiKey;
    if (this.cachedApiKey) return this.cachedApiKey;

    if (this.hostServices?.getConfig) {
      try {
        const apiKey = (
          await this.hostServices.getConfig("OPENROUTER_API_KEY")
        )?.trim();
        if (apiKey) {
          this.cachedApiKey = apiKey;
          return apiKey;
        }
      } catch {
        return undefined;
      }
    }
    return undefined;
  }

  // ── Mode Management ──────────────────────────────────────────────────

  getMode(): ProviderMode {
    return "sdk";
  }

  setMode(mode: ProviderMode): void {
    if (mode !== "sdk") {
      throw new Error(`${DISPLAY_NAME} only supports sdk mode`);
    }
  }

  private getAdapter(): ProviderAdapter {
    if (this.adapter) return this.adapter;
    this.adapter = new OpenRouterSdkAdapter(() => this.resolveAuth());
    this.log("info", `Adapter initialized in sdk mode`);
    return this.adapter;
  }

  // ── Session Management ───────────────────────────────────────────────

  async createSession(config: AISessionConfig): Promise<AISession> {
    const id =
      (config.providerConfig?.["sessionId"] as string) || crypto.randomUUID();
    const now = new Date().toISOString();

    const existing = this.sessions.get(id);
    if (existing) {
      existing.status = "active";
      existing.updatedAt = now;
      return this.toAISession(existing);
    }

    const session: ManagedSession = {
      id,
      config,
      status: "active",
      stats: {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      },
      abortController: null,
      files: [],
      createdAt: now,
      updatedAt: now,
    };

    this.sessions.set(id, session);
    this.log("info", `Session created: ${id} (${config.name})`);

    return this.toAISession(session);
  }

  async sendPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
  ): Promise<AIResponse> {
    const session = this.getSession(sessionId);
    session.status = "processing";
    session.updatedAt = new Date().toISOString();

    const abortController = new AbortController();
    session.abortController = abortController;

    const fullPrompt = this.buildFullPrompt(prompt, context, session.files);

    this.logIngester?.append({
      sessionId,
      type: "input",
      content: prompt,
    });

    try {
      const adapter = this.getAdapter();
      const result = await adapter.sendPrompt(
        fullPrompt,
        session.config,
        abortController.signal,
      );

      await this.updateSessionStats(
        session,
        result.inputTokens,
        result.outputTokens,
      );

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      });

      return {
        content: result.content,
        model: result.model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs: result.durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();

      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({
        sessionId,
        type: "error",
        content: errorMsg,
      });

      throw err;
    } finally {
      session.abortController = null;
    }
  }

  async streamPrompt(
    sessionId: string,
    prompt: string,
    context?: AIContext[],
    onChunk?: (chunk: AIStreamChunk) => void,
  ): Promise<AIResponse> {
    const session = this.getSession(sessionId);
    session.status = "processing";
    session.updatedAt = new Date().toISOString();

    const abortController = new AbortController();
    session.abortController = abortController;

    const fullPrompt = this.buildFullPrompt(prompt, context, session.files);

    this.logIngester?.append({
      sessionId,
      type: "input",
      content: prompt,
    });

    try {
      const adapter = this.getAdapter();
      const chunkHandler = onChunk ?? ((_c: AIStreamChunk) => {});

      const result = await adapter.streamPrompt(
        fullPrompt,
        session.config,
        chunkHandler,
        abortController.signal,
      );

      await this.updateSessionStats(
        session,
        result.inputTokens,
        result.outputTokens,
      );

      this.logIngester?.append({
        sessionId,
        type: "output",
        content: result.content,
        tokenCount: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      });

      return {
        content: result.content,
        model: result.model,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        durationMs: result.durationMs,
        metadata: result.metadata,
      };
    } catch (err) {
      session.status = "error";
      session.updatedAt = new Date().toISOString();

      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      this.logIngester?.append({
        sessionId,
        type: "error",
        content: errorMsg,
      });

      throw err;
    } finally {
      session.abortController = null;
    }
  }

  // ── Extended Methods ─────────────────────────────────────────────────

  async listModels(): Promise<AIModelInfo[]> {
    try {
      const models = await fetchOpenRouterModels();
      // Return a shallow copy so callers can mutate freely.
      return models.map((m) => ({ ...m }));
    } catch (err) {
      this.log(
        "error",
        `listModels failed: ${err instanceof Error ? err.message : String(err)}`,
      );
      return [];
    }
  }

  async cancelRequest(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);

    if (session.abortController) {
      session.abortController.abort();
      session.abortController = null;
      session.status = "active";
      session.updatedAt = new Date().toISOString();
      this.log("info", `Request cancelled for session: ${sessionId}`);
    }
  }

  getCapabilities(): AIProviderCapabilities {
    return {
      streaming: true,
      vision: true,
      fileAttachments: true,
      toolUse: true,
      mcpSupport: false,
      voiceMode: false,
      cancelSupport: true,
      modelListing: true,
    };
  }

  async attachFiles(
    sessionId: string,
    files: AIFileAttachment[],
  ): Promise<void> {
    const session = this.getSession(sessionId);
    session.files.push(...files);
    session.updatedAt = new Date().toISOString();
    this.log(
      "debug",
      `Attached ${files.length} file(s) to session ${sessionId}`,
    );
  }

  // ── Standard Methods ─────────────────────────────────────────────────

  async getSessionLogs(
    _sessionId: string,
    _filter?: AILogFilter,
  ): Promise<AILog[]> {
    return [];
  }

  async getUsageStats(sessionId: string): Promise<AIUsageStats> {
    const session = this.sessions.get(sessionId);
    return (
      session?.stats ?? {
        inputTokens: 0,
        outputTokens: 0,
        requestCount: 0,
        estimatedCostUsd: 0,
      }
    );
  }

  async configureSession(
    sessionId: string,
    config: Partial<AISessionConfig>,
  ): Promise<void> {
    const session = this.getSession(sessionId);
    Object.assign(session.config, config);
    session.updatedAt = new Date().toISOString();
  }

  async destroySession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (session) {
      if (session.abortController) {
        session.abortController.abort();
        session.abortController = null;
      }
      session.status = "terminated";
      session.files = [];
      session.updatedAt = new Date().toISOString();
      this.log("info", `Session terminated: ${sessionId}`);
    }
  }

  async listSessions(): Promise<AISession[]> {
    return Array.from(this.sessions.values()).map((s) => this.toAISession(s));
  }

  async getSessionStatus(sessionId: string): Promise<AISessionStatus> {
    return this.sessions.get(sessionId)?.status ?? "terminated";
  }

  async healthCheck(): Promise<{ ok: boolean; message?: string }> {
    const adapter = this.getAdapter();
    return adapter.healthCheck();
  }

  // ── `vibe ai run` / `vibe ai sdk` integration ────────────────────────

  /**
   * OpenRouter is SDK-only — there is no canonical CLI binary, so this
   * always returns null. `vibe ai run openrouter` will surface a helpful
   * error directing the user to `vibe ai sdk openrouter` instead.
   */
  getCliLaunchSpec(): {
    binary: string;
    baseArgs?: string[];
    env?: Record<string, string>;
  } | null {
    return null;
  }

  async sdkOneShot(opts: {
    prompt: string;
    model?: string;
    maxTokens?: number;
    extras?: Record<string, unknown>;
  }): Promise<{ text: string; usage?: unknown }> {
    const adapter = new OpenRouterSdkAdapter(() => this.resolveAuth());
    const config: AISessionConfig = {
      name: "vibe-ai-sdk",
      agentType: PROVIDER_NAME,
      model: opts.model ?? DEFAULT_MODEL,
      maxTokens: opts.maxTokens,
      providerConfig: opts.extras,
    };
    const result = await adapter.sendPrompt(opts.prompt, config);
    return {
      text: result.content,
      usage: {
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        model: result.model,
        durationMs: result.durationMs,
      },
    };
  }

  // ── Private Helpers ──────────────────────────────────────────────────

  private getSession(sessionId: string): ManagedSession {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    if (session.status === "terminated")
      throw new Error("Session is terminated");
    return session;
  }

  private buildFullPrompt(
    prompt: string,
    context?: AIContext[],
    files?: AIFileAttachment[],
  ): string {
    let fullPrompt = prompt;

    if (context && context.length > 0) {
      const contextStr = context
        .map((c) => `--- Context (${c.type}): ---\n${c.content}`)
        .join("\n\n");
      fullPrompt = `${prompt}\n\n${contextStr}`;
    }

    if (files && files.length > 0) {
      const fileStr = files
        .map((f) => {
          const textContent =
            typeof f.content === "string"
              ? f.content
              : f.content.toString("utf-8");
          return `--- File: ${f.filename} (${f.mimeType}, ${f.size} bytes) ---\n${textContent}`;
        })
        .join("\n\n");
      fullPrompt = `${fullPrompt}\n\n${fileStr}`;
    }

    return fullPrompt;
  }

  private async updateSessionStats(
    session: ManagedSession,
    inputTokens: number,
    outputTokens: number,
  ): Promise<void> {
    const model = session.config.model || DEFAULT_MODEL;

    session.stats.inputTokens += inputTokens;
    session.stats.outputTokens += outputTokens;
    session.stats.requestCount += 1;

    // Pricing comes from the live OpenRouter model list (cached). Skip cost
    // accounting silently if the cache hasn't been populated yet — listModels
    // will warm it on first frontend call.
    let modelInfo: AIModelInfo | undefined;
    try {
      const models = await fetchOpenRouterModels();
      modelInfo = models.find((m) => m.id === model);
    } catch {
      modelInfo = undefined;
    }

    if (modelInfo) {
      const cost =
        (inputTokens / 1_000_000) * modelInfo.inputPricePerMToken +
        (outputTokens / 1_000_000) * modelInfo.outputPricePerMToken;
      session.stats.estimatedCostUsd += cost;
    }

    if (!session.stats.modelBreakdown) {
      session.stats.modelBreakdown = {};
    }
    const breakdown = session.stats.modelBreakdown[model] ?? {
      inputTokens: 0,
      outputTokens: 0,
      requestCount: 0,
    };
    breakdown.inputTokens += inputTokens;
    breakdown.outputTokens += outputTokens;
    breakdown.requestCount += 1;
    session.stats.modelBreakdown[model] = breakdown;

    session.status = "active";
    session.updatedAt = new Date().toISOString();
  }

  private toAISession(s: ManagedSession): AISession {
    return {
      id: s.id,
      name: s.config.name,
      status: s.status,
      agentType: s.config.agentType,
      provider: PROVIDER_NAME,
      config: s.config,
      stats: s.stats,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
    };
  }

  private log(level: "info" | "error" | "debug", msg: string): void {
    this.logger?.[level](msg);
  }
}

// ── Plugin Export ────────────────────────────────────────────────────────

function createPrereqsRoutes() {
  // OpenRouter is SDK-only: no binary prerequisite, always satisfied.
  return new Elysia({ prefix: "/prereqs" })
    .get("/status", () => ({
      satisfied: true,
      missing: [],
    }))
    .post("/install", () => ({
      ok: true,
      installed: [],
      pendingSudo: [],
      errors: [],
    }));
}

const PLUGIN_NAME = "openrouter";
const PLUGIN_VERSION = "1.0.0";

const provider = new OpenRouterProvider();

const lifecycle = createLifecycleHooks({
  name: PLUGIN_NAME,
  telemetryEventName: "ai.provider.ready",
  onInit: (hostServices: HostServices) => {
    provider.setHostServices(hostServices);
    new TelemetryEmitter(PLUGIN_NAME, PLUGIN_VERSION, hostServices).emit(
      "ai.provider.ready",
      { provider: PLUGIN_NAME },
    );
  },
  onShutdown: () => {
    for (const [id] of (provider as OpenRouterProvider)["sessions"]) {
      provider.destroySession(id).catch(() => {});
    }
  },
});

type OpenRouterVibePlugin = VibePlugin & {
  providers?: { ai?: AIAgentProvider };
};

export const vibePlugin: OpenRouterVibePlugin = {
  capabilities: {
    secrets: "read",
    subprocess: true,
    gateway: false,
    telemetry: true,
  },
  name: PLUGIN_NAME,
  version: PLUGIN_VERSION,
  description: "OpenRouter AI agent provider for VibeControls (SDK)",
  tags: ["provider", "integration"],
  apiPrefix: API_PREFIX,
  prerequisites: [],
  providers: { ai: provider },
  createRoutes: () => createPrereqsRoutes(),
  onServerStart: lifecycle.onServerStart,
  onServerStop: lifecycle.onServerStop,
};

export default vibePlugin;
