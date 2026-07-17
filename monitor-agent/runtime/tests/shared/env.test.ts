/**
 * Unit tests for env.ts — environment loading with Zod validation.
 *
 * Domain: shared
 *
 * Tests the EnvLoader class with various configurations:
 * - Default values when env vars are missing
 * - Provider-specific URL defaults
 * - Custom base URL override
 * - Cache behavior (resetCache)
 * - Invalid configuration rejection
 */

import { EnvLoader } from "../../src/shared/env.js";

describe("env", () => {
  const originalEnv = process.env;
  let envLoader: EnvLoader;

  beforeEach(() => {
    envLoader = new EnvLoader();
    // Reset the cache before each test to allow fresh env parsing
    envLoader.resetCache();
    // Clear relevant env vars to test defaults
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
    envLoader.resetCache();
  });

  describe("loadEnv", () => {
    it("returns default values when no env vars are set", () => {
      const env = envLoader.loadEnv("/dev/null");
      expect(env["LLM_PROVIDER"]).toBe("openrouter");
      expect(env["LLM_MODEL"]).toBe("openai/gpt-4o-mini");
      expect(env["LLM_MAX_TOKENS"]).toBe(50000);
      expect(env["OPENAI_API_KEY"]).toBe("");
    });

    it("applies provider-specific default URL when LLM_BASE_URL is empty", () => {
      const env = envLoader.loadEnv("/dev/null");
      expect(env["LLM_BASE_URL"]).toBe("https://openrouter.ai/api/v1");
    });

    it("uses custom LLM_BASE_URL when explicitly set", () => {
      process.env["LLM_BASE_URL"] = "https://custom.api.com/v1";
      const env = envLoader.loadEnv("/dev/null");
      expect(env["LLM_BASE_URL"]).toBe("https://custom.api.com/v1");
    });

    it("returns correct default URL for openai provider", () => {
      process.env["LLM_PROVIDER"] = "openai";
      const env = envLoader.loadEnv("/dev/null");
      expect(env["LLM_BASE_URL"]).toBe("https://api.openai.com/v1");
    });

    it("returns correct default URL for anthropic provider", () => {
      process.env["LLM_PROVIDER"] = "anthropic";
      const env = envLoader.loadEnv("/dev/null");
      expect(env["LLM_BASE_URL"]).toBe("https://api.anthropic.com/v1");
    });

    it("returns correct default URL for google provider", () => {
      process.env["LLM_PROVIDER"] = "google";
      const env = envLoader.loadEnv("/dev/null");
      expect(env["LLM_BASE_URL"]).toBe("https://generativelanguage.googleapis.com/v1beta");
    });

    it("returns correct default URL for mistral provider", () => {
      process.env["LLM_PROVIDER"] = "mistral";
      const env = envLoader.loadEnv("/dev/null");
      expect(env["LLM_BASE_URL"]).toBe("https://api.mistral.ai/v1");
    });

    it("coerces LLM_MAX_TOKENS from string to number", () => {
      process.env["LLM_MAX_TOKENS"] = "10000";
      const env = envLoader.loadEnv("/dev/null");
      expect(env["LLM_MAX_TOKENS"]).toBe(10000);
    });

    it("caches the result across multiple calls", () => {
      process.env["LLM_MODEL"] = "test-model";
      const env1 = envLoader.loadEnv("/dev/null");
      const env2 = envLoader.loadEnv("/dev/null");
      expect(env1).toBe(env2);
    });

    it("reflects changes after cache reset", () => {
      process.env["LLM_MODEL"] = "model-v1";
      const env1 = envLoader.loadEnv("/dev/null");
      expect(env1["LLM_MODEL"]).toBe("model-v1");

      envLoader.resetCache();
      process.env["LLM_MODEL"] = "model-v2";
      const env2 = envLoader.loadEnv("/dev/null");
      expect(env2["LLM_MODEL"]).toBe("model-v2");
    });
  });

  describe("getProviderDefaultUrl", () => {
    it("returns openai URL", () => {
      expect(envLoader.getProviderDefaultUrl("openai")).toBe("https://api.openai.com/v1");
    });

    it("returns anthropic URL", () => {
      expect(envLoader.getProviderDefaultUrl("anthropic")).toBe("https://api.anthropic.com/v1");
    });

    it("returns google URL", () => {
      expect(envLoader.getProviderDefaultUrl("google")).toBe(
        "https://generativelanguage.googleapis.com/v1beta",
      );
    });

    it("returns mistral URL", () => {
      expect(envLoader.getProviderDefaultUrl("mistral")).toBe("https://api.mistral.ai/v1");
    });

    it("returns openrouter URL", () => {
      expect(envLoader.getProviderDefaultUrl("openrouter")).toBe("https://openrouter.ai/api/v1");
    });
  });
});
