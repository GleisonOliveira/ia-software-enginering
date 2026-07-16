/**
 * Type definitions for the main execution cycle.
 *
 * Domain: core
 *
 * Defines CycleConfig (runtime execution parameters), HistoryEntry (per-step record
 * of perception, plan, action result, and evaluation), Evaluation (objective completion
 * check), and Trace (full execution log for persistence and replay).
 */

import type { ActionType, QualityRating, TokenUsage } from "../shared/shared.types.js";
import type { Plan } from "../planner/planner.types.js";
import type { ActionResult } from "../shared/shared.types.js";

/**
 * Configuration for a single execution run, parsed from CLI arguments.
 *
 * Passed to the cycle runner's run() function to start execution.
 *
 * Used by: CLI run command, cycle runner entry point.
 */
export interface CycleConfig {
  /** Path to the agent directory containing contract .md files. */
  readonly agentPath: string;
  /** User input text for the agent to process. */
  readonly input: string;
  /** Optional agent type override (defaults to contract's tipo field). */
  readonly mode?: string;
  /** Optional event context string for the execution. */
  readonly event?: string;
  /** Optional output file path for saving results. */
  readonly output?: string;
}

/**
 * Result of the evaluator checking whether the current step achieved the objective.
 *
 * Used by: evaluator, cycle runner termination check, HistoryEntry, Trace.
 */
export interface Evaluation {
  /** True if the evaluator determines the objective has been achieved. */
  readonly objetivoAlcancado: boolean;
  /** Human-readable explanation of the evaluation result. */
  readonly motivo: string;
  /** Quality rating of the output (completa, parcial, falha); undefined if not yet rated. */
  readonly qualidade: QualityRating | undefined;
  /** List of output validation problems found (empty if output is valid). */
  readonly problemasSaida: readonly string[];
}

/**
 * Complete record of a single cycle step, capturing perception, plan, action, and evaluation.
 *
 * Appended to AgentState.historico at the end of each step.
 * Used for context continuity in the planner and for trace persistence.
 *
 * Used by: AgentState.historico, planner history input, Trace, trace writer.
 */
export interface HistoryEntry {
  /** Step number (0-indexed). */
  readonly etapa: number;
  /** The perception prompt generated for this step. */
  readonly percepcao: string;
  /** The LLM-generated plan for this step. */
  readonly plano: Plan;
  /** The action result if a tool was executed; undefined for FINALIZAR/PERGUNTAR_USUARIO. */
  readonly resultadoAcao: ActionResult | undefined;
  /** The evaluator's assessment of this step's outcome. */
  readonly avaliacao: Evaluation;
}

/**
 * Complete execution trace persisted to trace.json after each run.
 *
 * Contains all data needed for replay, analysis, and audit.
 * Written by the cycle runner after the final step completes.
 *
 * Used by: cycle runner, trace writer, analyze command, replay command.
 */
export interface Trace {
  /** Unique trace identifier (UUID). */
  readonly traceId: string;
  /** Agent type used for this execution. */
  readonly tipoAgente: string;
  /** Original user input. */
  readonly entrada: string;
  /** Optional event context. */
  readonly evento: string | undefined;
  /** Total execution time in seconds. */
  readonly tempoTotalSegundos: number;
  /** Cumulative token usage across all LLM calls. */
  readonly tokensConsumidos: TokenUsage;
  /** Step-by-step execution history. */
  readonly etapas: readonly HistoryEntry[];
  /** Final summary or result text. */
  readonly resumo: string;
  /** Agent name from the agent contract. */
  readonly agente: string;
  /** Full telemetry event stream for debugging. */
  readonly telemetryStream: readonly import("../telemetry/telemetry.types.js").TelemetryEvent[];
  /** Audit-relevant events subset. */
  readonly auditLogs: readonly import("../telemetry/telemetry.types.js").TelemetryEvent[];
  /** Aggregated health metrics. */
  readonly healthMetrics: import("../telemetry/telemetry.types.js").HealthMetrics;
  /** Aggregated performance data. */
  readonly performanceData: import("../telemetry/telemetry.types.js").PerformanceData;
}

/**
 * Valid action types the planner can return, extracted from the full ActionType union.
 *
 * Provides a narrower type for the cycle runner's action dispatch switch statement.
 *
 * Used by: cycle runner action dispatch, circuit breaker.
 */
export type ValidActionType = Extract<ActionType, "CHAMAR_FERRAMENTA" | "FINALIZAR" | "PERGUNTAR_USUARIO">;
