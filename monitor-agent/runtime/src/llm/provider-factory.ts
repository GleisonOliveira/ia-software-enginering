/**
 * Factory for creating AI SDK model instances from LlmConfig.
 *
 * Domain: llm
 *
 * Maps LlmProvider identifiers to Vercel AI SDK factory functions.
 * Centralizes provider instantiation logic used by both LlmClient and
 * structured output helpers, avoiding code duplication.
 *
 * Used by: llm-client.ts, structured-output.ts.
 */

import type { LlmConfig } from "./llm.types.js";
import type { LanguageModel } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createMistral } from "@ai-sdk/mistral";

/**
 * Creates AI SDK model instances from provider configuration.
 *
 * Encapsulates the mapping from LlmProvider strings to their
 * corresponding AI SDK factory functions. OpenRouter uses the OpenAI
 * provider with a custom baseURL since OpenRouter exposes an
 * OpenAI-compatible API.
 *
 * Used by: LlmClient.callLlm(), StructuredOutputHandler.
 */
export class ProviderFactory {
  /**
   * Creates an AI SDK LanguageModel for the given provider configuration.
   *
   * @param config - The resolved LLM configuration.
   * @returns An AI SDK LanguageModel instance ready for generateText/generateObject.
   *
   * Used by: LlmClient.callLlm(), generateStructuredOutput().
   */
  create(config: LlmConfig): LanguageModel {
    const providerSettings = {
      apiKey: config.apiKey,
      baseURL: config.baseURL,
    };

    switch (config.provider) {
      case "openai":
        return createOpenAI(providerSettings)(config.model);
      case "anthropic":
        return createAnthropic(providerSettings)(config.model);
      case "google":
        return createGoogleGenerativeAI(providerSettings)(config.model);
      case "mistral":
        return createMistral(providerSettings)(config.model);
      case "openrouter":
        // OpenRouter uses the OpenAI SDK with a custom baseURL
        return createOpenAI(providerSettings)(config.model);
      default: {
        // Exhaustive check — TypeScript will error if a case is missing
        const _exhaustive: never = config.provider;
        throw new Error(`Unsupported LLM provider: ${_exhaustive}`);
      }
    }
  }
}
