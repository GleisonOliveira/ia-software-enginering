/**
 * Structured output generation using Zod schema validation.
 *
 * Domain: llm
 *
 * Generates structured output from the LLM using a Zod schema constraint
 * via the AI SDK's generateObject(). Includes retry logic with Zod
 * validation error feedback for providers with lower structured output
 * reliability.
 *
 * Used by: planner module for Plan generation, tool builder for LLM-backed tools.
 */

import { generateObject } from "ai";
import { zodSchema } from "@ai-sdk/provider-utils";
import type { LlmConfig, StructuredOutputOptions, LlmUsage } from "./llm.types.js";
import type { TokenUsage } from "../shared/shared.types.js";
import { EMPTY_TOKEN_USAGE } from "../shared/shared.types.js";
import type { LlmConfigResolver } from "./llm-config.js";
import type { ProviderFactory } from "./provider-factory.js";
import type { Logger } from "../shared/logger.js";

/**
 * Maximum number of retry attempts when Zod validation fails.
 *
 * Why 3: The AI SDK's structured output mode handles schema translation
 * per provider, but edge cases (e.g., enum value mismatches) can still
 * occur. Three retries with error feedback is sufficient for self-correction.
 */
const MAX_RETRIES = 3;

/**
 * Result of a structured output generation call.
 *
 * Contains the parsed output object, token usage, and any retry metadata.
 * Used by the planner to receive type-safe LLM output.
 *
 * @typeParam T - The TypeScript type inferred from the Zod schema.
 */
export interface StructuredOutputResult<T> {
  /** The parsed and validated output matching the Zod schema. */
  readonly output: T;
  /** Token usage for this generation call (summed across retries). */
  readonly usage: TokenUsage;
  /** Number of retry attempts before success (0 = first attempt). */
  readonly retryCount: number;
}

/**
 * Interface for structured output attempt results including usage.
 */
interface AttemptResult<T> {
  output: T;
  usage: TokenUsage;
}

/**
 * Generates structured output from the LLM using Zod schema constraints.
 *
 * Attempts the AI SDK's generateObject() first, which uses provider-specific
 * structured output translation (OpenAI response_format, Anthropic output_config,
 * etc.). Falls back to text generation + manual Zod parsing when the
 * structured output call fails.
 *
 * Includes retry logic: when Zod validation fails, the validation error
 * is appended to the prompt and the LLM is asked to self-correct.
 *
 * Used by: planner callLlm() for Plan generation.
 */
export class StructuredOutputHandler {
  /** Resolves LLM config from environment variables. */
  private readonly configResolver: LlmConfigResolver;

  /** Creates AI SDK model instances from config. */
  private readonly providerFactory: ProviderFactory;

  /** Structured logger for debug and warning output. */
  private readonly logger: Logger;

  /**
   * @param configResolver - Resolves LLM config from environment variables.
   * @param providerFactory - Creates AI SDK model instances from config.
   * @param logger - Structured logger for debug and warning output.
   */
  constructor(
    configResolver: LlmConfigResolver,
    providerFactory: ProviderFactory,
    logger: Logger,
  ) {
    this.configResolver = configResolver;
    this.providerFactory = providerFactory;
    this.logger = logger;
  }

  /**
   * Generates a validated structured output from the LLM.
   *
   * @param options - Schema, prompts, and optional config overrides.
   * @returns The validated output, token usage, and retry count.
   * @throws Error if all retry attempts fail.
   *
   * Acceptance criteria:
   * - Validates LLM output against a Zod schema.
   * - Retry logic with Zod validation error feedback.
   */
  async generate<T>(
    options: StructuredOutputOptions<T>,
  ): Promise<StructuredOutputResult<T>> {
    const config = this.configResolver.resolve(options.config);
    const validationErrors = this.configResolver.validate(config);

    if (validationErrors.length > 0) {
      throw new Error(`Cannot generate structured output:\n${validationErrors.join("\n")}`);
    }

    let lastError: Error | undefined;
    let totalUsage: TokenUsage = { ...EMPTY_TOKEN_USAGE };

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        const result = await this.attemptGenerate<T>(config, options, attempt, totalUsage);
        return {
          output: result.output,
          usage: result.usage,
          retryCount: attempt,
        };
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        this.logger.warn(`Structured output attempt ${attempt + 1} failed`, {
          error: lastError.message,
        });

        if (error instanceof Error && "usage" in error) {
          const errUsage = (error as { usage?: LlmUsage }).usage;
          if (errUsage) {
            totalUsage = StructuredOutputHandler.accumulateUsage(totalUsage, errUsage);
          }
        }
      }
    }

    throw new Error(
      `Structured output failed after ${MAX_RETRIES} attempts. Last error: ${lastError?.message ?? "unknown"}`,
    );
  }

  /**
   * A single attempt at generating structured output.
   *
   * Uses the AI SDK's generateObject() for provider-native structured output.
   *
   * @param config - Resolved LLM configuration.
   * @param options - Original options (schema, prompts, overrides).
   * @param attempt - Current attempt number (0-based).
   * @param priorUsage - Accumulated token usage from prior attempts.
   * @returns The parsed output and usage for this attempt.
   */
  private async attemptGenerate<T>(
    config: LlmConfig,
    options: StructuredOutputOptions<T>,
    attempt: number,
    priorUsage: TokenUsage,
  ): Promise<AttemptResult<T>> {
    let userPrompt = options.prompt;
    if (attempt > 0) {
      userPrompt += `\n\n[RETRY] Previous attempt produced invalid output. Please follow the schema exactly.`;
    }

    const model = this.providerFactory.create(config);

    this.logger.debug("Generating structured output", {
      provider: config.provider,
      model: config.model,
      attempt: attempt + 1,
      maxRetries: MAX_RETRIES,
    });

    const result = await generateObject({
      model,
      schema: zodSchema(options.schema),
      system: options.systemPrompt,
      prompt: userPrompt,
    });

    const resultUsage = result.usage as { promptTokens?: number; completionTokens?: number; totalTokens?: number };
    const currentUsage: LlmUsage = {
      promptTokens: resultUsage?.promptTokens ?? 0,
      completionTokens: resultUsage?.completionTokens ?? 0,
      totalTokens: resultUsage?.totalTokens ?? 0,
    };

    const mergedUsage = StructuredOutputHandler.accumulateUsage(priorUsage, currentUsage);

    return {
      output: result.object as T,
      usage: mergedUsage,
    };
  }

  /**
   * Accumulates token usage from two calls by summing prompt, completion, and total.
   *
   * @param previous - Previous accumulated usage.
   * @param current - Current call's usage.
   * @returns Summed token usage.
   */
  private static accumulateUsage(previous: TokenUsage, current: LlmUsage): TokenUsage {
    return {
      prompt: previous.prompt + current.promptTokens,
      completion: previous.completion + current.completionTokens,
      total: previous.total + current.totalTokens,
    };
  }
}
