/**
 * Tool execution with retry logic.
 *
 * Domain: executor
 *
 * Executes registered tools by name with payload validation, retry on failure,
 * and token usage tracking. Wraps the tool function calls in a safe execution
 * context that catches errors and normalizes results.
 *
 * Used by: cycle runner (act phase), tool executor.
 */

import type { AllContracts } from "../contracts/contracts.types.js";
import type { ExecutionResult } from "./executor.types.js";
import type { ToolFunction } from "../tools/tools.types.js";
import type { ActionResult } from "../shared/shared.types.js";
import { EMPTY_TOKEN_USAGE } from "../shared/shared.types.js";
import type { PayloadValidator } from "./payload-validator.js";
import type { Logger } from "../shared/logger.js";

/**
 * Default number of retry attempts when tentar_novamente_em_falha is true.
 *
 * Why 1 (i.e., one retry = two total attempts): The Python runtime retries
 * exactly once on failure, which is sufficient for transient errors without
 * wasting tokens on persistent failures.
 */
const DEFAULT_RETRY_COUNT = 1;

/**
 * Executes tools by name with validation, retry, and token tracking.
 *
 * Performs the following steps:
 * 1. Validates the tool name exists in the tools map
 * 2. Optionally validates payload against the skill contract (if enabled)
 * 3. Executes the tool function
 * 4. On failure, optionally retries (if tentar_novamente_em_falha is true)
 * 5. Returns the normalized ExecutionResult with token usage
 *
 * Used by: cycle runner (act phase).
 */
export class ToolExecutor {
  /** Payload validator for pre-execution argument validation. */
  private readonly payloadValidator: PayloadValidator;

  /** Structured logger for debug and warning output. */
  private readonly logger: Logger;

  /**
   * @param payloadValidator - Validates tool arguments against skill schemas.
   * @param logger - Structured logger for debug and warning output.
   */
  constructor(payloadValidator: PayloadValidator, logger: Logger) {
    this.payloadValidator = payloadValidator;
    this.logger = logger;
  }

  /**
   * Executes a tool by name with validation, retry, and token tracking.
   *
   * @param toolName - Name of the tool to execute.
   * @param args - Arguments provided by the LLM planner.
   * @param tools - Map of tool names to their executable functions.
   * @param contracts - Full contract set for validation and retry config.
   * @returns The execution result with ActionResult and token usage.
   *
   * Used by: cycle runner (act phase).
   *
   * Acceptance criteria:
   * - Executes a tool and returns ActionResult.
   * - Retry logic from contracts.
   */
  async execute(
    toolName: string,
    args: Record<string, unknown>,
    tools: Map<string, ToolFunction>,
    contracts: AllContracts,
  ): Promise<ExecutionResult> {
    // 1. Check tool existence
    const toolFn = tools.get(toolName);
    if (!toolFn) {
      const availableTools = [...tools.keys()];
      const result: ActionResult = {
        sucesso: false,
        dados: {},
        erro: `Tool "${toolName}" not found. Available: ${availableTools.join(", ")}`,
        _tokens: EMPTY_TOKEN_USAGE,
        _entrada: args,
      };
      return { resultado: result, tokensUsados: EMPTY_TOKEN_USAGE };
    }

    // 2. Optionally validate payload before execution
    if (contracts.executor.execucao.validar_entrada) {
      const validationErrors = this.payloadValidator.validate(toolName, args, contracts);
      if (validationErrors.length > 0) {
        const result: ActionResult = {
          sucesso: false,
          dados: {},
          erro: `Payload validation failed: ${validationErrors.join("; ")}`,
          _tokens: EMPTY_TOKEN_USAGE,
          _entrada: args,
        };
        return { resultado: result, tokensUsados: EMPTY_TOKEN_USAGE };
      }
    }

    // 3. Execute the tool with optional retry
    const shouldRetry = contracts.executor.execucao.tentar_novamente_em_falha;
    const maxAttempts = shouldRetry ? DEFAULT_RETRY_COUNT + 1 : 1;

    let lastError: Error | undefined;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        this.logger.debug("Executing tool", { toolName, attempt: attempt + 1 });

        const resultado = await toolFn(args);

        this.logger.debug("Tool execution completed", {
          toolName,
          sucesso: resultado.sucesso,
        });

        return {
          resultado,
          tokensUsados: resultado._tokens ?? EMPTY_TOKEN_USAGE,
        };
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        this.logger.warn(`Tool execution failed (attempt ${attempt + 1}/${maxAttempts})`, {
          toolName,
          error: lastError.message,
        });
      }
    }

    // All attempts failed
    const result: ActionResult = {
      sucesso: false,
      dados: {},
      erro: lastError?.message ?? "Unknown execution error",
      _tokens: EMPTY_TOKEN_USAGE,
      _entrada: args,
    };

    return { resultado: result, tokensUsados: EMPTY_TOKEN_USAGE };
  }
}
