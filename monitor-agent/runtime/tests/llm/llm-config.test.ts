/**
 * Unit tests for llm-config.ts — LLM provider configuration resolution.
 *
 * Domain: llm
 *
 * Tests LlmConfigResolver.resolve() and validate() with various configurations:
 * - Default values from environment variables
 * - Provider-specific API key resolution
 * - Base URL resolution with overrides
 * - Config validation for missing fields
 */

import { describe, it, expect, beforeEach, afterEach } from "@jest/globals";
import { LlmConfigResolver, type EnvLoader } from "../../src/llm/llm-config.js";
import type { LlmEnv } from "../../src/shared/env.js";

/**
 * Creates a mock EnvLoader for testing.
 */
function createMockEnvLoader(overrides?: Partial<LlmEnv>): EnvLoader {
  return {
    loadEnv: (): LlmEnv => ({
      LLM_PROVIDER: "openrouter",
      LLM_BASE_URL: "",
      LLM_MODEL: "openai/gpt-4o-mini",
      LLM_MAX_TOKENS: 50000,
      OPENAI_API_KEY: "",
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

describe("LlmConfigResolver", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env["LLM_PROVIDER"];
    delete process.env["LLM_BASE_URL"];
    delete process.env["LLM_MODEL"];
    delete process.env["LLM_MAX_TOKENS"];
    delete process.env["OPENAI_API_KEY"];
    delete process.env["ANTHROPIC_API_KEY"];
    delete process.env["GOOGLE_GENERATIVE_AI_API_KEY"];
    delete process.env["MISTRAL_API_KEY"];
    delete process.env["OPENROUTER_API_KEY"];
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  describe("resolve", () => {
    it("returns default config from env loader", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader());
      const config = resolver.resolve();
      expect(config.provider).toBe("openrouter");
      expect(config.model).toBe("openai/gpt-4o-mini");
      expect(config.maxTokens).toBe(50000);
    });

    it("applies provider-specific default base URL", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader({ LLM_PROVIDER: "openai" }));
      const config = resolver.resolve();
      expect(config.baseURL).toBe("https://api.openai.com/v1");
    });

    it("uses custom base URL when set", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader({ LLM_BASE_URL: "https://custom.api.com/v1" }));
      const config = resolver.resolve();
      expect(config.baseURL).toBe("https://custom.api.com/v1");
    });

    it("resolves API key from provider-specific env var", () => {
      process.env["ANTHROPIC_API_KEY"] = "test-key-123";
      const resolver = new LlmConfigResolver(createMockEnvLoader({ LLM_PROVIDER: "anthropic" }));
      const config = resolver.resolve();
      expect(config.apiKey).toBe("test-key-123");
    });

    it("returns undefined API key when env var is empty", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader({ LLM_PROVIDER: "openai" }));
      const config = resolver.resolve();
      expect(config.apiKey).toBeUndefined();
    });

    it("applies overrides over env-derived values", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader());
      const config = resolver.resolve({
        model: "custom-model",
        maxTokens: 10000,
      });
      expect(config.model).toBe("custom-model");
      expect(config.maxTokens).toBe(10000);
    });

    it("override base URL takes precedence over env", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader({ LLM_BASE_URL: "https://env.api.com" }));
      const config = resolver.resolve({ baseURL: "https://override.api.com" });
      expect(config.baseURL).toBe("https://override.api.com");
    });
  });

  describe("validate", () => {
    it("returns empty array for valid config", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader());
      const errors = resolver.validate({
        provider: "openai",
        model: "gpt-4o-mini",
        baseURL: "https://api.openai.com/v1",
        maxTokens: 50000,
        apiKey: "test-key",
      });
      expect(errors).toHaveLength(0);
    });

    it("returns error when API key is missing", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader());
      const errors = resolver.validate({
        provider: "openai",
        model: "gpt-4o-mini",
        baseURL: "https://api.openai.com/v1",
        maxTokens: 50000,
        apiKey: undefined,
      });
      expect(errors.length).toBeGreaterThan(0);
      expect(errors[0]).toContain("OPENAI_API_KEY");
    });

    it("returns error when base URL is empty", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader());
      const errors = resolver.validate({
        provider: "openai",
        model: "gpt-4o-mini",
        baseURL: "",
        maxTokens: 50000,
        apiKey: "test-key",
      });
      expect(errors.length).toBeGreaterThan(0);
      expect(errors[0]).toContain("base URL");
    });

    it("returns error when model is empty", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader());
      const errors = resolver.validate({
        provider: "openai",
        model: "",
        baseURL: "https://api.openai.com/v1",
        maxTokens: 50000,
        apiKey: "test-key",
      });
      expect(errors.length).toBeGreaterThan(0);
      expect(errors[0]).toContain("model");
    });

    it("returns error when maxTokens is zero", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader());
      const errors = resolver.validate({
        provider: "openai",
        model: "gpt-4o-mini",
        baseURL: "https://api.openai.com/v1",
        maxTokens: 0,
        apiKey: "test-key",
      });
      expect(errors.length).toBeGreaterThan(0);
      expect(errors[0]).toContain("maxTokens");
    });

    it("returns multiple errors for multiple issues", () => {
      const resolver = new LlmConfigResolver(createMockEnvLoader());
      const errors = resolver.validate({
        provider: "openai",
        model: "",
        baseURL: "",
        maxTokens: 0,
        apiKey: undefined,
      });
      expect(errors.length).toBeGreaterThanOrEqual(3);
    });
  });
});
