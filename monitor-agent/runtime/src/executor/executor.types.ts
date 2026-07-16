/**
 * Type definitions for the executor domain.
 *
 * Domain: executor
 *
 * Defines ValidationResult (circuit breaker output), ExecutionResult (tool execution
 * outcome with token tracking), and ToolExecutorContext (execution context passed to
 * tool functions carrying tool name, arguments, and contract references).
 */

import type { ActionResult, TokenUsage } from "../shared/shared.types.js";
import type { AllContracts } from "../contracts/contracts.types.js";
import type { SkillParam } from "../contracts/contracts.types.js";

/**
 * Result of LLM response validation by the circuit breaker.
 *
 * Contains whether the plan is valid and a list of specific problems found.
 * An empty erros array indicates the plan passed all validation checks.
 *
 * Used by: circuit breaker, cycle runner (to decide whether to proceed or auto-correct).
 */
export interface ValidationResult {
  /** True if the LLM response passes all validation checks. */
  readonly valido: boolean;
  /** List of validation error messages (empty when valido is true). */
  readonly erros: readonly string[];
}

/**
 * Outcome of a tool execution, wrapping the ActionResult with token tracking.
 *
 * Used by: executor module, cycle runner, telemetry collector.
 */
export interface ExecutionResult {
  /** The tool's raw execution result. */
  readonly resultado: ActionResult;
  /** Token usage recorded during this execution (for LLM-backed tools). */
  readonly tokensUsados: TokenUsage;
}

/**
 * Execution context passed to tool functions during execution.
 *
 * Bundles tool name, arguments, and contract references so tools can
 * access everything they need without loose parameter passing.
 *
 * Used by: tool executor, tool builder, hook executor.
 */
export interface ToolExecutorContext {
  /** Name of the tool being executed. */
  readonly toolName: string;
  /** Arguments provided by the LLM planner. */
  readonly args: Record<string, unknown>;
  /** Full contract set for validation and reference. */
  readonly contracts: AllContracts;
}

/**
 * Schema mapping tool input parameter names to their type schemas.
 *
 * Used by: payload validator, tool builder, toolbox contract.
 */
export type ToolInputSchema = Record<string, SkillParam>;
