/**
 * Unit tests for llm-client.ts — provider-agnostic LLM client.
 *
 * Domain: llm
 *
 * Tests LlmClient class with mocked AI SDK:
 * - Successful text generation
 * - Configuration validation on construction
 */

import { jest } from "@jest/globals";
import type { LlmEnv } from "../../src/shared/env.js";
import type { EnvLoader } from "../../src/llm/llm-config.js";

jest.unstable_mockModule("ai", () => ({
  generateText: jest.fn(async () => ({
    text: "Hello from the LLM",
    usage: {
      promptTokens: 10,
      completionTokens: 20,
      totalTokens: 30,
    },
  })),
}));

const { LlmClient } = await import("../../src/llm/llm-client.js");
const { LlmConfigResolver } = await import("../../src/llm/llm-config.js");
const { ProviderFactory } = await import("../../src/llm/provider-factory.js");
const { Logger } = await import("../../src/shared/logger.js");
const ai = await import("ai");

function createMockEnvLoader(overrides?: { OPENAI_API_KEY?: string }): EnvLoader {
  return {
    loadEnv: (): LlmEnv => ({
      LLM_PROVIDER: "openai",
      LLM_BASE_URL: "https://api.openai.com/v1",
      LLM_MODEL: "gpt-4o-mini",
      LLM_MAX_TOKENS: 4096,
      OPENAI_API_KEY: overrides?.OPENAI_API_KEY ?? "test-key-123",
      ANTHROPIC_API_KEY: "",
      GOOGLE_GENERATIVE_AI_API_KEY: "",
      MISTRAL_API_KEY: "",
      OPENROUTER_API_KEY: "",
    }),
    getProviderDefaultUrl: (provider: string): string => {
      const urls: Record<string, string> = {
        openai: "https://api.openai.com/v1",
        anthropic: "https://api.anthropic.com/v1",
        google: "https://generativelanguage.googleapis.com/v1beta",
        mistral: "https://api.mistral.ai/v1",
        openrouter: "https://openrouter.ai/api/v1",
      };
      return urls[provider] ?? "";
    },
  };
}

describe("LlmClient", () => {
  const originalEnv = process.env;
  const envLoader = createMockEnvLoader();
  const configResolver = new LlmConfigResolver(envLoader);
  const providerFactory = new ProviderFactory();
  const logger = new Logger("error");

  beforeEach(() => {
    process.env = { ...originalEnv };
    process.env["OPENAI_API_KEY"] = "test-key-123";
    jest.mocked(ai.generateText).mockClear();
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  describe("construction", () => {
    it("creates client with valid config", () => {
      const client = new LlmClient(configResolver, providerFactory, logger);
      expect(client).toBeDefined();
      expect(client.getConfig().provider).toBe("openai");
    });

    it("throws when API key is missing", () => {
      delete process.env["OPENAI_API_KEY"];
      const resolver = new LlmConfigResolver(envLoader);
      expect(() => new LlmClient(resolver, providerFactory, logger)).toThrow("configuration errors");
    });
  });

  describe("callLlm", () => {
    it("returns text and usage from LLM", async () => {
      const client = new LlmClient(configResolver, providerFactory, logger);
      const response = await client.callLlm({
        systemPrompt: "You are a test assistant",
        userPrompt: "Say hello",
      });
      expect(response.text).toBe("Hello from the LLM");
      expect(response.usage.promptTokens).toBe(10);
      expect(response.usage.completionTokens).toBe(20);
      expect(response.usage.totalTokens).toBe(30);
    });
  });
});
