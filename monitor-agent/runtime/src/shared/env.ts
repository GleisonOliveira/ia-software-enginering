/**
 * Environment variable loading with Zod validation.
 *
 * Domain: shared
 *
 * Loads .env via dotenv and validates all LLM configuration variables
 * using Zod schemas. Provides provider-specific defaults for LLM_BASE_URL
 * when the variable is empty, enabling seamless provider switching.
 *
 * Used by: LLM client, tool builder, planner, cycle runner.
 */

import { z } from "zod";
import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Zod schema for the provider identifier.
 *
 * Maps to the AI SDK provider packages installed in the project.
 * Each value corresponds to a @ai-sdk/* package.
 */
const ProviderSchema = z.enum(["openai", "anthropic", "google", "mistral", "openrouter"]);

/**
 * Zod schema for the complete LLM environment configuration.
 *
 * Validates all environment variables used by the runtime. Defaults are
 * applied when variables are missing or empty, matching the .env.example.
 */
const LlmEnvSchema = z.object({
  /** The active LLM provider (openai, anthropic, google, mistral, openrouter). */
  LLM_PROVIDER: ProviderSchema.default("openrouter"),
  /** Custom base URL for the provider API; empty triggers provider-specific defaults. */
  LLM_BASE_URL: z.string().default(""),
  /** Model identifier string (provider/model format for OpenRouter). */
  LLM_MODEL: z.string().default("openai/gpt-4o-mini"),
  /** Maximum tokens the LLM may consume across all calls in a single execution. */
  LLM_MAX_TOKENS: z.coerce.number().int().positive().default(50000),
  /** API key for OpenAI; required when using the openai provider. */
  OPENAI_API_KEY: z.string().default(""),
  /** API key for Anthropic; required when using the anthropic provider. */
  ANTHROPIC_API_KEY: z.string().default(""),
  /** API key for Google Generative AI; required when using the google provider. */
  GOOGLE_GENERATIVE_AI_API_KEY: z.string().default(""),
  /** API key for Mistral; required when using the mistral provider. */
  MISTRAL_API_KEY: z.string().default(""),
  /** API key for OpenRouter; required when using the openrouter provider. */
  OPENROUTER_API_KEY: z.string().default(""),
});

/**
 * Inferred TypeScript type for the validated environment configuration.
 *
 * Ensures compile-time type safety across all env-consuming modules.
 */
export type LlmEnv = z.infer<typeof LlmEnvSchema>;

/**
 * Provider-specific default base URLs.
 *
 * Each provider has a different API endpoint. When LLM_BASE_URL is empty,
 * the appropriate default is selected based on LLM_PROVIDER. This allows
 * providers to work out of the box without explicit URL configuration.
 */
const PROVIDER_DEFAULT_URLS: Record<z.infer<typeof ProviderSchema>, string> = {
  openai: "https://api.openai.com/v1",
  anthropic: "https://api.anthropic.com/v1",
  google: "https://generativelanguage.googleapis.com/v1beta",
  mistral: "https://api.mistral.ai/v1",
  openrouter: "https://openrouter.ai/api/v1",
};

/**
 * Loaded environment cache to avoid re-reading process.env on every call.
 *
 * Set once during loadEnv() and immutable thereafter. This prevents
 * race conditions if loadEnv were called concurrently (unlikely but safe).
 */
let cachedEnv: LlmEnv | undefined;

/**
 * Loads and validates environment variables using dotenv and Zod.
 *
 * Reads .env from the project root, applies provider-specific defaults
 * for LLM_BASE_URL when empty, and returns the validated configuration.
 * Caches the result so subsequent calls return the same object.
 *
 * @param envPath - Optional explicit path to the .env file.
 *   When omitted, auto-detects from the project root.
 * @returns The validated and typed environment configuration.
 *
 * Used by: LLM client initialization, tool builder, cycle runner startup.
 */
export function loadEnv(envPath?: string): LlmEnv {
  if (cachedEnv) return cachedEnv;

  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);

  // Resolve .env path: explicit > project root > current directory
  const resolvedPath = envPath ?? path.resolve(__dirname, "../../.env");
  dotenv.config({ path: resolvedPath });

  // Validate process.env through Zod, applying defaults for missing values
  const parsed = LlmEnvSchema.safeParse(process.env);

  if (!parsed.success) {
    const issues = parsed.error.issues.map((i) => `  ${i.path.join(".")}: ${i.message}`).join("\n");
    throw new Error(`Invalid environment configuration:\n${issues}`);
  }

  const env = parsed.data;

  // Apply provider-specific default URL when LLM_BASE_URL is empty
  if (env.LLM_BASE_URL === "") {
    env.LLM_BASE_URL = PROVIDER_DEFAULT_URLS[env.LLM_PROVIDER];
  }

  cachedEnv = env;
  return cachedEnv;
}

/**
 * Returns the provider-specific default base URL for a given provider.
 *
 * Useful when explicitly constructing a provider client without
 * relying on loadEnv() caching behavior.
 *
 * @param provider - The LLM provider identifier.
 * @returns The default API base URL for that provider.
 *
 * Used by: LLM provider config resolution.
 */
export function getProviderDefaultUrl(provider: z.infer<typeof ProviderSchema>): string {
  return PROVIDER_DEFAULT_URLS[provider];
}

/**
 * Clears the cached environment configuration.
 *
 * Intended for testing only — allows re-reading environment variables
 * after mocking process.env in test suites.
 *
 * Used by: Unit tests that need to test different env configurations.
 */
export function resetEnvCache(): void {
  cachedEnv = undefined;
}
