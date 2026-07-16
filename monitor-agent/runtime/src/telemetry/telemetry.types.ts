/**
 * Type definitions for the telemetry domain.
 *
 * Domain: telemetry
 *
 * Defines TelemetryEvent (timestamped execution event), PhaseMarker (per-phase
 * timing data), PhaseStats (aggregated timing statistics), HealthMetrics (tool
 * success rates and circuit breaker activations), and PerformanceData (execution
 * performance summary). Used for trace persistence and the post-run KPI panel.
 */

import type { TokenUsage } from "../shared/shared.types.js";

/**
 * Categorizes telemetry events by their type in the execution lifecycle.
 *
 * Each event type corresponds to a specific point in the cycle or a
 * significant state change. Used for filtering and aggregation in the
 * post-run KPI panel.
 *
 * Used by: TelemetryEvent.tipo, telemetry collector, audit log filtering.
 */
export type TelemetryEventType =
  /** Cycle started. */
  | "inicio"
  /** LLM generated a plan. */
  | "plano_gerado"
  /** A cycle phase (perceive/plan/act/evaluate) completed. */
  | "fase_concluida"
  /** A tool was executed. */
  | "ferramenta_executada"
  /** Circuit breaker detected an invalid plan. */
  | "circuit_breaker"
  /** Payload validation failed for a tool call. */
  | "validacao_payload_falha"
  /** Human confirmation was requested for a sensitive action. */
  | "confirmacao_humana"
  /** Time limit was exceeded. */
  | "limite_tempo_excedido"
  /** Token limit was exceeded. */
  | "limite_tokens_excedido"
  /** Cycle completed (objective achieved or limits hit). */
  | "finalizado";

/**
 * Identifies a specific phase within the perceive→plan→act→evaluate cycle.
 *
 * Used by: PhaseMarker.fase, performance data aggregation.
 */
export type PhaseName =
  /** Perception phase: building the context prompt. */
  | "perceber"
  /** Planning phase: LLM generates the next action. */
  | "planejar"
  /** Action phase: executing the chosen tool. */
  | "agir"
  /** Evaluation phase: checking if the objective was achieved. */
  | "avaliar"
  /** Payload validation phase: checking tool input arguments. */
  | "validar_payload";

/**
 * A single timestamped telemetry event recorded during execution.
 *
 * Events are appended to the trace stream and used for debugging,
 * audit logging, and the post-run KPI panel.
 *
 * Used by: Telemetry collector, trace writer, audit log, health metrics.
 */
export interface TelemetryEvent {
  /** ISO 8601 timestamp of the event. */
  readonly timestamp: string;
  /** Milliseconds elapsed since cycle start. */
  readonly elapsedMs: number;
  /** Unique identifier for this execution run. */
  readonly traceId: string;
  /** Event type categorizing this event. */
  readonly tipo: TelemetryEventType;
  /** Event-specific data payload (structure varies by tipo). */
  readonly dados: Record<string, unknown>;
}

/**
 * Timing data for a single phase execution within a cycle step.
 *
 * Used by: Telemetry collector, performance data aggregation.
 */
export interface PhaseMarker {
  /** Phase name (perceber, planejar, agir, avaliar, validar_payload). */
  readonly fase: PhaseName;
  /** Step number this phase belongs to. */
  readonly etapa: number;
  /** Start timestamp in milliseconds (from performance.now()). */
  readonly inicio: number;
  /** End timestamp in milliseconds; undefined if phase is still running. */
  readonly fim: number | undefined;
  /** Duration in milliseconds; undefined if phase is still running. */
  readonly duracaoMs: number | undefined;
}

/**
 * Aggregated timing statistics for a specific phase across all steps.
 *
 * Used by: PerformanceData.fases, post-run KPI panel.
 */
export interface PhaseStats {
  /** Total time spent in this phase across all steps (ms). */
  readonly totalMs: number;
  /** Number of times this phase was executed. */
  readonly contagem: number;
  /** Maximum single-phase duration (ms). */
  readonly maxMs: number;
  /** Average phase duration (ms). */
  readonly mediaMs: number;
}

/**
 * Health metrics summarizing tool execution success rates and error counts.
 *
 * Used by: Telemetry collector, post-run KPI panel, trace persistence.
 */
export interface HealthMetrics {
  /** Execution trace identifier. */
  readonly traceId: string;
  /** Percentage of tool calls that succeeded (0-1). */
  readonly taxaSucessoFerramentas: number;
  /** Total successful tool calls. */
  readonly ferramentasSucesso: number;
  /** Total failed tool calls. */
  readonly ferramentasFalha: number;
  /** Number of circuit breaker activations (invalid LLM plans). */
  readonly circuitBreakerAtivacoes: number;
  /** Number of payload validation failures. */
  readonly validacaoPayloadFalhas: number;
  /** Total LLM API calls made. */
  readonly chamadasLlm: number;
}

/**
 * Execution performance summary with timing, token, and phase breakdown.
 *
 * Used by: Telemetry collector, post-run KPI panel, trace persistence.
 */
export interface PerformanceData {
  /** Execution trace identifier. */
  readonly traceId: string;
  /** Total execution time in milliseconds. */
  readonly tempoTotalMs: number;
  /** Cumulative token usage across all LLM calls. */
  readonly tokens: TokenUsage;
  /** Total LLM API calls made. */
  readonly chamadasLlm: number;
  /** Per-phase timing statistics keyed by phase name. */
  readonly fases: Readonly<Record<string, PhaseStats>>;
}

/**
 * Complete telemetry summary for a single execution run.
 *
 * Combines all telemetry data into a single object for persistence
 * and post-run analysis.
 *
 * Used by: trace writer, analyze command, replay command.
 */
export interface TelemetrySummary {
  /** Unique execution trace identifier. */
  readonly traceId: string;
  /** Agent name from the agent contract. */
  readonly agente: string;
  /** Agent type (task_based, interactive, etc.). */
  readonly tipoAgente: string;
  /** Full stream of telemetry events recorded during execution. */
  readonly telemetryStream: readonly TelemetryEvent[];
  /** Audit-relevant events (subset of telemetryStream). */
  readonly auditLogs: readonly TelemetryEvent[];
  /** Aggregated health metrics. */
  readonly healthMetrics: HealthMetrics;
  /** Aggregated performance data. */
  readonly performanceData: PerformanceData;
}
