/**
 * Provider-agnostic LLM client wrapping Vercel AI SDK's generateText.
 *
 * Domain: llm
 *
 * Provides a single callLlm() method that delegates to the AI SDK's
 * generateText() with the resolved provider configuration. Handles
 * provider instantiation, token usage extraction, and error normalization.
 * Supports all providers via the unified AI SDK interface.
 *
 * Used by: planner module, structured output helpers, tool builder.
 */

import { generateText } from "ai";
import type { LlmConfig, CallLlmOptions, LlmResponse, LlmUsage } from "./llm.types.js";
import type { LlmConfigResolver } from "./llm-config.js";
import type { ProviderFactory } from "./provider-factory.js";
import type { Logger } from "../shared/logger.js";

/**
 * Extracts token usage from the AI SDK response.
 *
 * The AI SDK returns usage in a slightly different shape than our
 * TokenUsage interface. This normalizes the fields and handles
 * undefined usage (which can happen with some providers).
 *
 * @param usage - The raw usage from the AI SDK response.
 * @returns Normalized LlmUsage with all fields guaranteed to be numbers.
 *
 * Used by: LlmClient.callLlm().
 */
function extractUsage(usage: { promptTokens?: number; completionTokens?: number; totalTokens?: number } | undefined): LlmUsage {
  return {
    promptTokens: usage?.promptTokens ?? 0,
    completionTokens: usage?.completionTokens ?? 0,
    totalTokens: usage?.totalTokens ?? 0,
  };
}

/**
 * Provider-agnostic LLM client for text generation.
 *
 * Wraps Vercel AI SDK's generateText() with injected configuration.
 * The provider, model, and token limits are resolved via LlmConfigResolver
 * and can be overridden per call.
 *
 * Why: The Python runtime was hardcoded to OpenAI. This class uses the AI SDK
 * to support multiple providers via env vars without code changes.
 *
 * Used by: planner callLlm(), mockPlanner(), tool builder for LLM-backed tools.
 */
export class LlmClient {
  /** Resolved LLM configuration (provider, model, API key, limits). */
  private readonly config: LlmConfig;

  /** Factory for creating AI SDK model instances. */
  private readonly providerFactory: ProviderFactory;

  /** Structured logger for debug and error output. */
  private readonly logger: Logger;

  /**
   * Creates an LlmClient with injected dependencies.
   *
   * Validates the configuration before construction to fail fast
   * on missing API keys or invalid settings.
   *
   * @param configResolver - Resolves LLM config from environment variables.
   * @param providerFactory - Creates AI SDK model instances from config.
   * @param logger - Structured logger for debug and error output.
   * @param overrides - Optional partial config to override env-derived values.
   * @throws Error if configuration validation fails.
   */
  constructor(
    configResolver: LlmConfigResolver,
    providerFactory: ProviderFactory,
    logger: Logger,
    overrides?: Partial<LlmConfig>,
  ) {
    this.providerFactory = providerFactory;
    this.logger = logger;
    this.config = configResolver.resolve(overrides);

    const validationErrors = configResolver.validate(this.config);
    if (validationErrors.length > 0) {
      throw new Error(`LlmClient configuration errors:\n${validationErrors.join("\n")}`);
    }
  }

  /**
   * Calls the LLM for text generation with the resolved configuration.
   *
   * Delegates to the AI SDK's generateText() with the system and user prompts.
   * Extracts token usage from the response and returns a normalized LlmResponse.
   *
   * @param options - System prompt, user prompt, and optional config overrides.
   * @returns The LLM's text response and token usage breakdown.
   *
   * Used by: planner module for plan generation, tool builder for LLM-backed tools.
   *
   * Acceptance criteria:
   * - Returns text and TokenUsage with configurable provider via env.
   */
  async callLlm(options: CallLlmOptions): Promise<LlmResponse> {
    const config = options.config
      ? { ...this.config, ...options.config }
      : this.config;

    const model = this.providerFactory.create(config);

    this.logger.debug("Calling LLM", {
      provider: config.provider,
      model: config.model,
      systemPromptLength: options.systemPrompt.length,
      userPromptLength: options.userPrompt.length,
    });

    const result = await generateText({
      model,
      system: options.systemPrompt,
      prompt: options.userPrompt,
      maxOutputTokens: config.maxTokens,
    });

    const usage = extractUsage(result.usage);

    this.logger.debug("LLM response received", {
      textLength: result.text.length,
      promptTokens: usage.promptTokens,
      completionTokens: usage.completionTokens,
      totalTokens: usage.totalTokens,
    });

    return {
      text: result.text,
      usage,
    };
  }

  /**
   * Returns the resolved configuration for this client instance.
   *
   * Useful for logging and debugging which provider/model is active.
   *
   * @returns A readonly copy of the resolved LlmConfig.
   */
  getConfig(): Readonly<LlmConfig> {
    return this.config;
  }
}
