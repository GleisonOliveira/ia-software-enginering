/**
 * Type definitions for the planner domain.
 *
 * Domain: planner
 *
 * Defines Plan (the structured LLM output describing the next action), Perception
 * (string alias for the perception prompt), and PlannerContext (bundled inputs
 * for the planner: perception, system prompt, and execution history).
 */

import type { ActionType } from "../shared/shared.types.js";

/**
 * Structured LLM output describing the next action to take.
 *
 * Returned by the planner after processing the current perception and history.
 * The circuit breaker validates this before the cycle runner acts on it.
 *
 * Used by: circuit breaker, cycle runner action dispatch, executor, evaluator,
 * HistoryEntry, Trace.
 */
export interface Plan {
  /** The action type: CHAMAR_FERRAMENTA, FINALIZAR, or PERGUNTAR_USUARIO. */
  readonly proximaAcao: ActionType;
  /** Tool name when proximaAcao is CHAMAR_FERRAMENTA; undefined otherwise. */
  readonly nomeFerramenta: string | undefined;
  /** Tool arguments when proximaAcao is CHAMAR_FERRAMENTA; undefined otherwise. */
  readonly argumentosFerramenta: Record<string, unknown> | undefined;
  /** Success criterion describing when this step's action is considered complete. */
  readonly criterioSucesso: string;
  /** Question for the user when proximaAcao is PERGUNTAR_USUARIO; undefined otherwise. */
  readonly pergunta: string | undefined;
}

/**
 * Type alias for the perception prompt string.
 *
 * A plain string, but aliased for semantic clarity across planner interfaces.
 *
 * Used by: PlannerContext, perception builder.
 */
export type Perception = string;

/**
 * Bundled inputs for the planner module.
 *
 * Groups perception, system prompt, and history into a single object
 * for cleaner function signatures in the planner.
 *
 * Used by: planner callLlm(), mockPlanner().
 */
export interface PlannerContext {
  /** The current perception prompt describing the agent's state. */
  readonly perception: Perception;
  /** The system prompt built from contracts defining LLM behavior. */
  readonly systemPrompt: string;
  /** Execution history from previous steps (for context continuity). */
  readonly history: readonly import("../core/cycle.types.js").HistoryEntry[];
}
