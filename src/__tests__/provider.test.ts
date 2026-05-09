/**
 * vibe-plugin-ai-openrouter Provider Tests
 *
 * Tests for the OpenRouterProvider class exported via the vibePlugin.
 */
import { beforeAll, beforeEach, describe, expect, it, mock } from "bun:test";

// Mock @openrouter/sdk before importing the plugin
mock.module("@openrouter/sdk", () => {
  class MockOpenRouter {
    chat = {
      send: mock(
        (req: { chatRequest: { stream?: boolean; model: string } }) => {
          if (req.chatRequest.stream) {
            // Async iterable streaming chunks
            const chunks = [
              {
                choices: [{ delta: { content: "streamed " } }],
                model: req.chatRequest.model,
              },
              {
                choices: [{ delta: { content: "text" } }],
                model: req.chatRequest.model,
                usage: { promptTokens: 5, completionTokens: 15 },
              },
            ];
            return Promise.resolve({
              async *[Symbol.asyncIterator]() {
                for (const c of chunks) yield c;
              },
            });
          }
          return Promise.resolve({
            choices: [{ message: { content: "Hello from OpenRouter" } }],
            model: req.chatRequest.model,
            usage: { promptTokens: 10, completionTokens: 20 },
          });
        },
      ),
    };
  }
  return { OpenRouter: MockOpenRouter, default: MockOpenRouter };
});

// Stub the global fetch used for the live model list so listModels()
// doesn't hit the network during tests.
beforeAll(() => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = mock(async (url: string | URL | Request) => {
    const u = typeof url === "string" ? url : url.toString();
    if (u.includes("/api/v1/models")) {
      return new Response(
        JSON.stringify({
          data: [
            {
              id: "openai/gpt-4o-mini",
              name: "GPT-4o Mini",
              context_length: 128_000,
              top_provider: { max_completion_tokens: 16_384 },
              architecture: { input_modalities: ["text", "image"] },
              pricing: { prompt: "0.00000015", completion: "0.0000006" },
            },
            {
              id: "anthropic/claude-3.5-sonnet",
              name: "Claude 3.5 Sonnet",
              context_length: 200_000,
              top_provider: { max_completion_tokens: 8192 },
              architecture: { input_modalities: ["text", "image"] },
              pricing: { prompt: "0.000003", completion: "0.000015" },
            },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }
    return originalFetch(url);
  }) as unknown as typeof fetch;
});

const { createPlugin } = await import("../index.js");
const vibePlugin = createPlugin({ name: "test", dataDir: "/tmp" });

const provider = vibePlugin.providers!.ai!;

describe("OpenRouterProvider", () => {
  const sessionConfig = {
    name: "test-session",
    agentType: "openrouter",
    model: "openai/gpt-4o-mini",
    maxTokens: 4096,
  };

  beforeEach(() => {
    process.env["OPENROUTER_API_KEY"] = "test-key-123";
  });

  // ── Session Lifecycle ───────────────────────────────────────────

  describe("createSession", () => {
    it("creates a new session with generated ID", async () => {
      const session = await provider.createSession({ ...sessionConfig });

      expect(session.id).toBeDefined();
      expect(session.name).toBe("test-session");
      expect(session.agentType).toBe("openrouter");
      expect(session.provider).toBe("openrouter");
      expect(session.status).toBe("active");
      expect(session.stats.inputTokens).toBe(0);
      expect(session.createdAt).toBeDefined();
    });

    it("uses provided sessionId from providerConfig", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "custom-id-001" },
      });
      expect(session.id).toBe("custom-id-001");
    });

    it("returns existing session if ID already exists", async () => {
      const session1 = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "reuse-id" },
      });
      const session2 = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "reuse-id" },
      });
      expect(session1.id).toBe(session2.id);
      expect(session2.status).toBe("active");
    });
  });

  describe("configureSession", () => {
    it("updates session config", async () => {
      const session = await provider.createSession({ ...sessionConfig });
      await provider.configureSession(session.id, {
        model: "anthropic/claude-3.5-sonnet",
      });

      const sessions = await provider.listSessions();
      const updated = sessions.find((s) => s.id === session.id);
      expect(updated?.config.model).toBe("anthropic/claude-3.5-sonnet");
    });

    it("throws for non-existent session", async () => {
      await expect(
        provider.configureSession("does-not-exist", { model: "x" }),
      ).rejects.toThrow("not found");
    });
  });

  describe("destroySession", () => {
    it("terminates session and cleans up", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: "destroy-me" },
      });
      await provider.destroySession(session.id);
      const status = await provider.getSessionStatus(session.id);
      expect(status).toBe("terminated");
    });

    it("no-ops for unknown session ID", async () => {
      await provider.destroySession("nonexistent-session");
    });
  });

  describe("listSessions", () => {
    it("returns all sessions", async () => {
      const id = `list-test-${Date.now()}`;
      await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: id },
      });
      const sessions = await provider.listSessions();
      expect(sessions.length).toBeGreaterThanOrEqual(1);
      expect(sessions.some((s) => s.id === id)).toBe(true);
    });
  });

  describe("getSessionStatus", () => {
    it("returns status for existing session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `status-${Date.now()}` },
      });
      const status = await provider.getSessionStatus(session.id);
      expect(status).toBe("active");
    });

    it("returns terminated for unknown session", async () => {
      const status = await provider.getSessionStatus("totally-unknown");
      expect(status).toBe("terminated");
    });
  });

  // ── sendPrompt ──────────────────────────────────────────────────

  describe("sendPrompt", () => {
    it("sends prompt via SDK adapter and returns response", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `prompt-${Date.now()}` },
      });
      const response = await provider.sendPrompt(session.id, "What is 2+2?");

      expect(response.content).toBe("Hello from OpenRouter");
      expect(response.model).toBe("openai/gpt-4o-mini");
      expect(response.inputTokens).toBe(10);
      expect(response.outputTokens).toBe(20);
      expect(response.durationMs).toBeGreaterThanOrEqual(0);
    });

    it("accumulates usage stats across multiple prompts", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `multi-prompt-${Date.now()}` },
      });
      await provider.sendPrompt(session.id, "First prompt");
      await provider.sendPrompt(session.id, "Second prompt");
      const stats = await provider.getUsageStats(session.id);
      expect(stats.inputTokens).toBe(20);
      expect(stats.outputTokens).toBe(40);
      expect(stats.requestCount).toBe(2);
    });

    it("throws for non-existent session", async () => {
      await expect(provider.sendPrompt("ghost", "Hello")).rejects.toThrow(
        "not found",
      );
    });

    it("throws for terminated session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `terminated-prompt-${Date.now()}` },
      });
      await provider.destroySession(session.id);
      await expect(provider.sendPrompt(session.id, "Hello")).rejects.toThrow(
        "terminated",
      );
    });
  });

  // ── getUsageStats ───────────────────────────────────────────────

  describe("getUsageStats", () => {
    it("returns zero stats for fresh session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `fresh-stats-${Date.now()}` },
      });
      const stats = await provider.getUsageStats(session.id);
      expect(stats.inputTokens).toBe(0);
      expect(stats.outputTokens).toBe(0);
      expect(stats.requestCount).toBe(0);
      expect(stats.estimatedCostUsd).toBe(0);
    });

    it("returns default stats for unknown session", async () => {
      const stats = await provider.getUsageStats("no-such-session");
      expect(stats.inputTokens).toBe(0);
      expect(stats.requestCount).toBe(0);
    });
  });

  // ── healthCheck ─────────────────────────────────────────────────

  describe("healthCheck", () => {
    it("returns ok when SDK is available", async () => {
      const result = await provider.healthCheck();
      expect(result.ok).toBe(true);
      expect(result.message).toContain("SDK");
    });
  });

  // ── getCapabilities ─────────────────────────────────────────────

  describe("getCapabilities", () => {
    it("returns SDK-mode capabilities", () => {
      const caps = provider.getCapabilities!();
      expect(caps.streaming).toBe(true);
      expect(caps.vision).toBe(true);
      expect(caps.fileAttachments).toBe(true);
      expect(caps.toolUse).toBe(true);
      expect(caps.cancelSupport).toBe(true);
      expect(caps.modelListing).toBe(true);
    });
  });

  // ── getMode / setMode ───────────────────────────────────────────

  describe("getMode / setMode", () => {
    it("always reports sdk mode", () => {
      expect(provider.getMode!()).toBe("sdk");
    });

    it("rejects non-sdk modes", () => {
      expect(() => provider.setMode!("cli")).toThrow();
    });

    it("accepts sdk mode setter as a no-op", () => {
      provider.setMode!("sdk");
      expect(provider.getMode!()).toBe("sdk");
    });
  });

  // ── listModels ──────────────────────────────────────────────────

  describe("listModels", () => {
    it("returns models from the live OpenRouter API", async () => {
      const models = await provider.listModels!();
      expect(models.length).toBeGreaterThanOrEqual(2);
      expect(models.every((m) => m.provider === "openrouter")).toBe(true);

      const sonnet = models.find((m) => m.id.includes("claude-3.5-sonnet"));
      expect(sonnet).toBeDefined();
      expect(sonnet!.contextWindow).toBe(200_000);
      expect(sonnet!.supportsVision).toBe(true);
      // Pricing converted from per-token to per-million.
      expect(sonnet!.inputPricePerMToken).toBeCloseTo(3, 5);
      expect(sonnet!.outputPricePerMToken).toBeCloseTo(15, 5);
    });

    it("returns a copy each call", async () => {
      const a = await provider.listModels!();
      const b = await provider.listModels!();
      expect(a).not.toBe(b);
      expect(a).toEqual(b);
    });
  });

  // ── cancelRequest ───────────────────────────────────────────────

  describe("cancelRequest", () => {
    it("throws for unknown session", async () => {
      await expect(provider.cancelRequest!("missing")).rejects.toThrow(
        "not found",
      );
    });
  });

  // ── attachFiles ─────────────────────────────────────────────────

  describe("attachFiles", () => {
    it("attaches files to an existing session", async () => {
      const session = await provider.createSession({
        ...sessionConfig,
        providerConfig: { sessionId: `files-${Date.now()}` },
      });
      await provider.attachFiles!(session.id, [
        {
          filename: "test.txt",
          mimeType: "text/plain",
          content: "hello",
          size: 5,
        },
      ]);
    });

    it("throws for non-existent session", async () => {
      await expect(
        provider.attachFiles!("none", [
          { filename: "f.txt", mimeType: "text/plain", content: "x", size: 1 },
        ]),
      ).rejects.toThrow("not found");
    });
  });
});
