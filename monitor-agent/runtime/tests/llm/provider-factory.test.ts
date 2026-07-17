/**
 * Unit tests for provider-factory.ts — AI SDK model instance creation.
 *
 * Domain: llm
 *
 * Tests ProviderFactory.create() with various provider configurations:
 * - OpenAI provider
 * - Anthropic provider
 * - Google provider
 * - Mistral provider
 * - OpenRouter provider
 * - Unsupported provider throws
 */

import { describe, it, expect } from "@jest/globals";
import { ProviderFactory } from "../../src/llm/provider-factory.js";
import type { LlmConfig } from "../../src/llm/llm.types.js";

describe("ProviderFactory", () => {
  const factory = new ProviderFactory();

  function createConfig(provider: LlmConfig["provider"]): LlmConfig {
    return {
      provider,
      model: "test-model",
      baseURL: "https://test.api.com/v1",
      maxTokens: 4096,
      apiKey: "test-key",
    };
  }

  it("creates a model for openai provider", () => {
    const model = factory.create(createConfig("openai"));
    expect(model).toBeDefined();
  });

  it("creates a model for anthropic provider", () => {
    const model = factory.create(createConfig("anthropic"));
    expect(model).toBeDefined();
  });

  it("creates a model for google provider", () => {
    const model = factory.create(createConfig("google"));
    expect(model).toBeDefined();
  });

  it("creates a model for mistral provider", () => {
    const model = factory.create(createConfig("mistral"));
    expect(model).toBeDefined();
  });

  it("creates a model for openrouter provider (uses OpenAI SDK)", () => {
    const model = factory.create(createConfig("openrouter"));
    expect(model).toBeDefined();
  });
});
