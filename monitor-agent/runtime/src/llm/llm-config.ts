/**
 * LLM provider configuration resolution from environment variables.
 *
 * Domain: llm
 *
 * Resolves the active LLM provider, model, API key, base URL, and token
 * limits from environment variables. Maps provider identifiers to Vercel
 * AI SDK factory functions and selects the correct API key environment
 * variable per provider.
 *
 * Used by: LlmClient constructor, StructuredOutputHandler.
 */

import type { LlmProvider, LlmConfig } from "./llm.types.js";
import type { LlmEnv } from "../shared/env.js";

/**
 * Interface for environment loading dependencies.
 *
 * Abstracts the env module so LlmConfigResolver can be tested
 * with a mock loader and doesn't depend on module-level singletons.
 */
export interface EnvLoader {
  loadEnv(): LlmEnv;
  getProviderDefaultUrl(provider: LlmProvider): string;
}

/**
 * Maps each LLM provider to the environment variable holding its API key.
 *
 * When resolving a provider, the corresponding env var is read. If empty,
 * the provider will still be constructed (the AI SDK handles missing keys
 * with a descriptive error at call time).
 */
const PROVIDER_API_KEY_VARS: Record<LlmProvider, string> = {
  openai: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
  google: "GOOGLE_GENERATIVE_AI_API_KEY",
  mistral: "MISTRAL_API_KEY",
  openrouter: "OPENROUTER_API_KEY",
};

/**
 * Resolves LlmConfig from validated environment variables.
 *
 * Reads the environment via the injected EnvLoader and maps each field to
 * the LlmConfig interface. The API key is resolved from the provider-specific
 * env var (e.g., OPENAI_API_KEY for the "openai" provider). When the base
 * URL is empty, the provider-specific default is applied.
 *
 * Used by: LlmClient constructor, StructuredOutputHandler.
 */
export class LlmConfigResolver {
  /** Injected environment loader for reading and validating env vars. */
  private readonly envLoader: EnvLoader;

  /**
   * @param envLoader - Injected environment loader (provides loadEnv() and getProviderDefaultUrl()).
   */
  constructor(envLoader: EnvLoader) {
    this.envLoader = envLoader;
  }

  /**
   * Resolves the full LlmConfig from the validated environment.
   *
   * @param overrides - Optional partial config to merge over env-derived values.
   * @returns The fully resolved LlmConfig.
   *
   * Acceptance criteria:
   * - Selects the correct provider based on LLM_PROVIDER.
   * - Support provider-specific env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
   * - LLM_BASE_URL env var — custom endpoint for OpenRouter, local models, proxies.
   * - Model selection from LLM_MODEL env var.
   * - Token limits from LLM_MAX_TOKENS env var.
   */
  resolve(overrides?: Partial<LlmConfig>): LlmConfig {
    const env = this.envLoader.loadEnv();

    const provider: LlmProvider = env.LLM_PROVIDER;

    const apiKeyVar = PROVIDER_API_KEY_VARS[provider];
    const apiKey = process.env[apiKeyVar] ?? undefined;

    const baseURL = overrides?.baseURL ?? (env.LLM_BASE_URL || this.envLoader.getProviderDefaultUrl(provider));

    return {
      provider,
      model: overrides?.model ?? env.LLM_MODEL,
      baseURL,
      maxTokens: overrides?.maxTokens ?? env.LLM_MAX_TOKENS,
      apiKey: overrides?.apiKey ?? apiKey,
    };
  }

  /**
   * Validates that the resolved config has all required fields for a provider.
   *
   * Returns a list of missing fields rather than throwing, so callers can
   * provide a user-friendly error message.
   *
   * @param config - The LlmConfig to validate.
   * @returns Array of validation error messages (empty if config is valid).
   */
  validate(config: LlmConfig): string[] {
    const errors: string[] = [];

    if (!config.apiKey) {
      const envVar = PROVIDER_API_KEY_VARS[config.provider];
      errors.push(`Missing API key for provider "${config.provider}". Set ${envVar} environment variable.`);
    }

    if (!config.baseURL) {
      errors.push(`Missing base URL for provider "${config.provider}".`);
    }

    if (!config.model) {
      errors.push("Missing model identifier. Set LLM_MODEL environment variable.");
    }

    if (config.maxTokens <= 0) {
      errors.push(`Invalid maxTokens value: ${config.maxTokens}. Must be a positive integer.`);
    }

    return errors;
  }
}
