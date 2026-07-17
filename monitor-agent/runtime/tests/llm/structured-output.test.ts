/**
 * Unit tests for structured-output.ts — structured output generation.
 *
 * Domain: llm
 *
 * Tests StructuredOutputHandler with mocked AI SDK:
 * - Configuration validation
 * - Error on invalid config
 */

import { describe, it, expect } from "@jest/globals";
import { StructuredOutputHandler } from "../../src/llm/structured-output.js";
import { LlmConfigResolver, type EnvLoader } from "../../src/llm/llm-config.js";
import { ProviderFactory } from "../../src/llm/provider-factory.js";
import { Logger } from "../../src/shared/logger.js";
import { z } from "zod";
import type { LlmEnv } from "../../src/shared/env.js";

/**
 * Creates a mock EnvLoader with valid defaults.
 */
function createMockEnvLoader(overrides?: Partial<LlmEnv>): EnvLoader {
  return {
    loadEnv: (): LlmEnv => ({
      LLM_PROVIDER: "openai",
      LLM_BASE_URL: "https://api.openai.com/v1",
      LLM_MODEL: "gpt-4o-mini",
      LLM_MAX_TOKENS: 4096,
      OPENAI_API_KEY: "test-key-123",
      ANTHROPIC_API_KEY: "",
      GOOGLE_GENERATIVE_AI_API_KEY: "",
      MISTRAL_API_KEY: "",
      OPENROUTER_API_KEY: "",
      ...overrides,
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

describe("StructuredOutputHandler", () => {
  const schema = z.object({
    name: z.string(),
    value: z.number(),
  });

  describe("construction", () => {
    it("creates handler with valid config", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader());
      const factory = new ProviderFactory();
      const logger = new Logger("error");
      const handler = new StructuredOutputHandler(resolver, factory, logger);
      expect(handler).toBeDefined();
    });
  });

  describe("generate", () => {
    it("throws when config validation fails for overrides", async () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader());
      const factory = new ProviderFactory();
      const logger = new Logger("error");
      const handler = new StructuredOutputHandler(resolver, factory, logger);

      await expect(
        handler.generate({
          schema,
          systemPrompt: "You are a test",
          prompt: "Generate data",
          config: { apiKey: undefined, baseURL: "" },
        }),
      ).rejects.toThrow("Cannot generate structured output");
    });
  });
});
