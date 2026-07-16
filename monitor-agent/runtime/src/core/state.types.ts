/**
 * Type definitions for the agent runtime state.
 *
 * Domain: core
 *
 * Defines AgentState (the mutable state object passed through the perceive→plan→act→evaluate
 * cycle) and StateLimits (step/token/time limits loaded from contracts). Also provides
 * a MutableAgentState utility type for internal state mutations while keeping the public
 * interface immutable.
 */

import type { AgentType } from "../shared/shared.types.js";
import type { TokenUsage } from "../shared/shared.types.js";
import type { HistoryEntry } from "./cycle.types.js";

/**
 * Resource limits loaded from the regras.md contract.
 *
 * Extracted once at state creation and used by the cycle runner to enforce
 * step, token, time, and per-tool call limits throughout execution.
 *
 * Used by: AgentState.limits, cycle runner limit checks.
 */
export interface StateLimits {
  /** Maximum number of cycle steps before forced termination. */
  readonly maxEtapas: number;
  /** Maximum total tool calls across all tools. */
  readonly maxChamadasFerramenta: number;
  /** Per-tool call limits: tool name -> max calls. */
  readonly limitesPorFerramenta: Readonly<Record<string, number>>;
  /** Steps without progress before stagnation detection triggers. */
  readonly semProgresso: number;
  /** Total execution time limit in seconds. */
  readonly limiteTempoSegundos: number;
  /** Maximum tokens the LLM may consume across all calls. */
  readonly maxTokens: number;
}

/**
 * The mutable state object passed through the perceive→plan→act→evaluate cycle.
 *
 * Created once at cycle start via createState() and updated at each step.
 * The public interface is readonly; internal mutations use MutableAgentState.
 *
 * Used by: cycle runner, perception builder, planner, executor, evaluator,
 * telemetry collector, trace writer.
 */
export interface AgentState {
  /** The high-level objective from the agent contract. */
  readonly objetivo: string;
  /** User-provided input text for this execution run. */
  readonly entrada: string;
  /** Agent type controlling cycle behavior (task_based, interactive, etc.). */
  readonly tipoAgente: AgentType;
  /** Optional event context for the current execution. */
  readonly evento: string | undefined;
  /** Current step number (0-indexed, incremented each cycle). */
  readonly etapa: number;
  /** Total tool calls made so far across all tools. */
  readonly chamadasFerramenta: number;
  /** Per-tool call counts: tool name -> calls made. */
  readonly chamadasPorFerramenta: Record<string, number>;
  /** Resource limits loaded from the regras.md contract. */
  readonly limits: StateLimits;
  /** Cumulative token usage across all LLM calls. */
  readonly tokensConsumidos: TokenUsage;
  /** Actions requiring human confirmation (from regras.md acoes_sensiveis). */
  readonly acoesSensiveis: readonly string[];
  /** Execution history: one entry per completed step. */
  readonly historico: readonly HistoryEntry[];
  /** Whether the objective has been achieved (cycle terminates when true). */
  readonly concluido: boolean;
  /** Final output text from FINALIZAR action. */
  readonly resultado: string;
  /** Consecutive steps without progress (incremented when evaluator detects stagnation). */
  readonly etapasSemProgresso: number;
  /** Name of the most recently executed tool (used for stagnation detection). */
  readonly ultimaFerramenta: string | undefined;
}

/**
 * Utility type that removes readonly modifiers from AgentState.
 *
 * Used internally by the cycle runner and state manager to perform mutations
 * while the public interface remains immutable. Converts readonly arrays to
 * mutable arrays and readonly records to mutable records.
 *
 * Used by: state manager, cycle runner internal logic.
 */
export type MutableAgentState = {
  -readonly [K in keyof AgentState]: AgentState[K] extends Record<string, unknown>
    ? MutableAgentState[K]
    : AgentState[K] extends readonly (infer U)[]
      ? U[]
      : AgentState[K];
};
