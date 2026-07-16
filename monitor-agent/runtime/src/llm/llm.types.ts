/**
 * Type definitions for the LLM provider abstraction layer.
 *
 * Domain: llm
 *
 * Defines provider-agnostic interfaces (LlmConfig, CallLlmOptions, LlmResponse, etc.)
 * consumed by the LLM client and planner modules. The concrete provider selection
 * is resolved at runtime via environment variables, keeping these types decoupled
 * from any specific AI SDK implementation.
 */

import type { z } from "zod";

/**
 * Supported LLM provider identifiers.
 *
 * Maps to Vercel AI SDK provider modules (@ai-sdk/openai, @ai-sdk/anthropic, etc.).
 * OpenRouter uses the openai provider with a custom baseURL.
 *
 * Used by: LlmConfig, resolveProvider(), environment variable LLM_PROVIDER.
 */
export type LlmProvider = "openai" | "anthropic" | "google" | "mistral" | "openrouter";

/**
 * Configuration for an LLM provider instance.
 *
 * Resolved at runtime from environment variables by resolveProvider().
 * All fields are readonly to prevent accidental mutation after construction.
 *
 * Used by: LlmClient constructor, CallLlmOptions, StructuredOutputOptions.
 */
export interface LlmConfig {
  /** Provider identifier selecting the AI SDK module. */
  readonly provider: LlmProvider;
  /** Model string (e.g., "gpt-4o-mini", "claude-3-haiku"). */
  readonly model: string;
  /** Custom base URL for proxies, OpenRouter, or local models; undefined uses provider default. */
  readonly baseURL: string | undefined;
  /** Maximum tokens the LLM may generate in a single response. */
  readonly maxTokens: number;
  /** API key for the provider; resolved from provider-specific env vars. */
  readonly apiKey: string | undefined;
}

/**
 * Options for a single LLM text generation call.
 *
 * Used by: LlmClient.callLlm(), planner module.
 */
export interface CallLlmOptions {
  /** System prompt setting the LLM's behavior and constraints. */
  readonly systemPrompt: string;
  /** User prompt containing the perception and task description. */
  readonly userPrompt: string;
  /** Optional config overrides merged with the default LlmConfig. */
  readonly config?: Partial<LlmConfig>;
}

/**
 * Response from a text-generation LLM call.
 *
 * Used by: planner, structured output helpers.
 */
export interface LlmResponse {
  /** Raw text output from the LLM. */
  readonly text: string;
  /** Token usage breakdown for this call. */
  readonly usage: LlmUsage;
}

/**
 * Token usage breakdown from an LLM API response.
 *
 * Maps to the usage field returned by Vercel AI SDK's generateText().
 */
export interface LlmUsage {
  /** Number of tokens in the input prompt. */
  readonly promptTokens: number;
  /** Number of tokens in the generated completion. */
  readonly completionTokens: number;
  /** Total tokens consumed (prompt + completion). */
  readonly totalTokens: number;
}

/**
 * Options for structured output generation using Zod schema validation.
 *
 * Used by: generateStructuredOutput() helper, planner structured output calls.
 * The AI SDK's Output.object() handles provider-specific schema translation.
 *
 * @typeParam T - The TypeScript type inferred from the Zod schema.
 */
export interface StructuredOutputOptions<T> {
  /** Zod schema defining the expected output structure. */
  readonly schema: z.ZodType<T>;
  /** User prompt requesting the structured output. */
  readonly prompt: string;
  /** System prompt guiding the LLM's structured response format. */
  readonly systemPrompt: string;
  /** Optional config overrides merged with the default LlmConfig. */
  readonly config?: Partial<LlmConfig>;
}
