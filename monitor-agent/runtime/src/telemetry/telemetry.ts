/**
 * Telemetry collector — records execution events, timing, and metrics.
 *
 * Domain: telemetry
 *
 * Collects structured telemetry data during a single agent execution run:
 * - Timestamped events for debugging and audit logging
 * - Per-phase timing data for performance analysis
 * - Token usage aggregation across all LLM calls
 * - Tool success/failure counters for health metrics
 * - Circuit breaker and payload validation failure tracking
 *
 * Mirrors the Python runtime's Telemetria class.
 *
 * Used by: cycle runner (all phases), trace writer, KPI panel.
 */

import { v4 as uuidv4 } from "uuid";
import type {
  TelemetryEvent,
  TelemetryEventType,
  PhaseMarker,
  PhaseName,
  PhaseStats,
  HealthMetrics,
  PerformanceData,
  TelemetrySummary,
} from "./telemetry.types.js";
import type { TokenUsage } from "../shared/shared.types.js";
import { EMPTY_TOKEN_USAGE } from "../shared/shared.types.js";

/**
 * Audit-relevant event types that appear in the audit log stream.
 *
 * Subset of TelemetryEventType used for filtering events that represent
 * decisions or actions relevant to compliance auditing.
 */
const AUDIT_EVENT_TYPES = new Set<TelemetryEventType>([
  "plano_gerado",
  "ferramenta_executada",
  "circuit_breaker",
  "validacao_payload_falha",
  "confirmacao_humana",
  "finalizado",
]);

/**
 * Collects and aggregates telemetry data for a single execution run.
 *
 * Created once at cycle start with the agent name and type. Provides
 * methods for recording events, tracking phase timing, accumulating
 * token usage, and computing health/performance metrics.
 *
 * Used by: cycle runner, trace writer, KPI panel.
 */
export class Telemetry {
  /** Unique trace identifier for this execution run. */
  readonly traceId: string;

  /** Agent name from the agent contract. */
  readonly agente: string;

  /** Agent type (task_based, interactive, etc.). */
  readonly tipoAgente: string;

  /** Start time of the execution (performance.now() value). */
  private readonly inicio: number;

  /** Accumulated telemetry events. */
  private readonly eventos: TelemetryEvent[] = [];

  /** Phase markers for completed phases. */
  private readonly fases: PhaseMarker[] = [];

  /** Accumulated token usage across all LLM calls. */
  private tokens: TokenUsage = { ...EMPTY_TOKEN_USAGE };

  /** Total LLM API calls made. */
  private chamadasLlm = 0;

  /** Successful tool execution count. */
  private ferramentasSucesso = 0;

  /** Failed tool execution count. */
  private ferramentasFalha = 0;

  /** Circuit breaker activation count. */
  private circuitBreakerAtivacoes = 0;

  /** Payload validation failure count. */
  private validacaoPayloadFalhas = 0;

  /**
   * Creates a new Telemetry collector for an execution run.
   *
   * @param agente - Agent name from the contract.
   * @param tipoAgente - Agent type controlling cycle behavior.
   */
  constructor(agente: string, tipoAgente: string) {
    this.traceId = uuidv4().replace(/-/g, "").substring(0, 12);
    this.agente = agente;
    this.tipoAgente = tipoAgente;
    this.inicio = performance.now();
  }

  /**
   * Records a telemetry event with timestamp and trace ID.
   *
   * @param tipo - The event type categorizing this event.
   * @param dados - Event-specific data payload.
   *
   * Used by: cycle runner, circuit breaker, payload validator.
   */
  registrar(tipo: TelemetryEventType, dados: Record<string, unknown> = {}): void {
    this.eventos.push({
      timestamp: new Date().toISOString(),
      elapsedMs: Math.round(performance.now() - this.inicio),
      traceId: this.traceId,
      tipo,
      dados,
    });
  }

  /**
   * Starts phase timing measurement.
   *
   * Returns a PhaseMarker that must be passed to finalizarFase() to
   * complete the measurement. The marker tracks start time and phase metadata.
   *
   * @param fase - Phase name (perceber, planejar, agir, avaliar, validar_payload).
   * @param etapa - Current step number.
   * @returns A PhaseMarker with the start timestamp.
   *
   * Used by: cycle runner (all phases).
   */
  iniciarFase(fase: PhaseName, etapa: number): PhaseMarker {
    return {
      fase,
      etapa,
      inicio: performance.now(),
      fim: undefined,
      duracaoMs: undefined,
    };
  }

  /**
   * Finalizes phase timing and records the completed marker.
   *
   * Calculates the duration and records a fase_concluida event.
   *
   * @param marcador - The PhaseMarker returned by iniciarFase().
   *
   * Used by: cycle runner (all phases).
   */
  finalizarFase(marcador: PhaseMarker): void {
    const fim = performance.now();
    const duracaoMs = Math.round((fim - marcador.inicio) * 100) / 100;

    // Create a completed marker (TypeScript doesn't allow mutation of readonly fields)
    const completedMarker: PhaseMarker = {
      fase: marcador.fase,
      etapa: marcador.etapa,
      inicio: marcador.inicio,
      fim,
      duracaoMs,
    };

    this.fases.push(completedMarker);

    this.registrar("fase_concluida", {
      fase: marcador.fase,
      etapa: marcador.etapa,
      duracaoMs,
    });
  }

  /**
   * Accumulates token usage from an LLM call.
   *
   * @param uso - Token usage from the LLM response.
   *
   * Used by: cycle runner after LLM calls.
   */
  registrarTokens(uso: TokenUsage): void {
    this.tokens = {
      prompt: this.tokens.prompt + uso.prompt,
      completion: this.tokens.completion + uso.completion,
      total: this.tokens.total + uso.total,
    };
    this.chamadasLlm += 1;
  }

  /**
   * Records a tool execution success or failure.
   *
   * @param sucesso - Whether the tool execution succeeded.
   *
   * Used by: cycle runner (act phase).
   */
  registrarResultadoFerramenta(sucesso: boolean): void {
    if (sucesso) {
      this.ferramentasSucesso += 1;
    } else {
      this.ferramentasFalha += 1;
    }
  }

  /**
   * Records a circuit breaker activation.
   *
   * @param motivo - Description of why the circuit breaker fired.
   *
   * Used by: cycle runner (circuit breaker validation).
   */
  registrarCircuitBreaker(motivo: string): void {
    this.circuitBreakerAtivacoes += 1;
    this.registrar("circuit_breaker", { motivo });
  }

  /**
   * Records a payload validation failure.
   *
   * @param ferramenta - Name of the tool that failed validation.
   * @param erros - List of validation error messages.
   *
   * Used by: cycle runner (payload validation phase).
   */
  registrarValidacaoPayloadFalha(ferramenta: string, erros: string[]): void {
    this.validacaoPayloadFalhas += 1;
    this.registrar("validacao_payload_falha", { ferramenta, erros });
  }

  /**
   * Returns the complete telemetry event stream.
   *
   * @returns Array of all recorded telemetry events.
   *
   * Used by: trace writer, analyze command.
   */
  telemetryStream(): readonly TelemetryEvent[] {
    return this.eventos;
  }

  /**
   * Returns audit-relevant events (decisions and actions).
   *
   * Filters the full event stream to only include events that
   * represent decisions or actions relevant to compliance auditing.
   *
   * @returns Array of audit-relevant events.
   *
   * Used by: trace writer, audit log analysis.
   */
  auditLogs(): readonly TelemetryEvent[] {
    return this.eventos.filter((e) => AUDIT_EVENT_TYPES.has(e.tipo));
  }

  /**
   * Returns health metrics summarizing tool execution success rates.
   *
   * @returns HealthMetrics with success rate, counters, and activation counts.
   *
   * Used by: trace writer, post-run KPI panel.
   */
  healthMetrics(): HealthMetrics {
    const totalFerramentas = this.ferramentasSucesso + this.ferramentasFalha;
    const taxaSucesso = totalFerramentas > 0
      ? Math.round((this.ferramentasSucesso / totalFerramentas) * 1000) / 10
      : 0;

    return {
      traceId: this.traceId,
      taxaSucessoFerramentas: taxaSucesso,
      ferramentasSucesso: this.ferramentasSucesso,
      ferramentasFalha: this.ferramentasFalha,
      circuitBreakerAtivacoes: this.circuitBreakerAtivacoes,
      validacaoPayloadFalhas: this.validacaoPayloadFalhas,
      chamadasLlm: this.chamadasLlm,
    };
  }

  /**
   * Returns performance data with per-phase timing and token usage.
   *
   * Aggregates phase markers into per-phase statistics (total, count,
   * max, avg) and includes cumulative token usage.
   *
   * @returns PerformanceData with timing, tokens, and phase breakdown.
   *
   * Used by: trace writer, post-run KPI panel.
   */
  performanceData(): PerformanceData {
    const fases: Record<string, { totalMs: number; contagem: number; maxMs: number; mediaMs: number }> = {};

    for (const fase of this.fases) {
      const nome = fase.fase;
      if (!fases[nome]) {
        fases[nome] = { totalMs: 0, contagem: 0, maxMs: 0, mediaMs: 0 };
      }

      const stats = fases[nome];
      if (fase.duracaoMs !== undefined) {
        stats.totalMs += fase.duracaoMs;
        stats.contagem += 1;
        if (fase.duracaoMs > stats.maxMs) {
          stats.maxMs = fase.duracaoMs;
        }
      }
    }

    // Calculate averages
    for (const stats of Object.values(fases)) {
      stats.mediaMs = stats.contagem > 0
        ? Math.round((stats.totalMs / stats.contagem) * 10) / 10
        : 0;
    }

    return {
      traceId: this.traceId,
      tempoTotalMs: Math.round(performance.now() - this.inicio),
      tokens: { ...this.tokens },
      chamadasLlm: this.chamadasLlm,
      fases: fases as Readonly<Record<string, PhaseStats>>,
    };
  }

  /**
   * Returns latencies for a specific step (planejar and agir phases).
   *
   * Used by the KPI panel to display per-step timing.
   *
   * @param etapa - The step number to get latencies for.
   * @returns Map of phase name to duration in ms.
   *
   * Used by: cycle runner KPI display.
   */
  kpisEtapa(etapa: number): Record<string, number> {
    const latencias: Record<string, number> = {};

    for (const fase of this.fases) {
      if (fase.etapa === etapa && (fase.fase === "planejar" || fase.fase === "agir")) {
        if (fase.duracaoMs !== undefined) {
          latencias[fase.fase] = fase.duracaoMs;
        }
      }
    }

    return latencias;
  }

  /**
   * Returns the complete telemetry summary for trace persistence.
   *
   * Combines all telemetry data into a single TelemetrySummary object
   * suitable for writing to trace.json.
   *
   * @returns Complete telemetry summary.
   *
   * Used by: trace writer, cycle runner finalization.
   */
  resumoCompleto(): TelemetrySummary {
    return {
      traceId: this.traceId,
      agente: this.agente,
      tipoAgente: this.tipoAgente,
      telemetryStream: this.telemetryStream(),
      auditLogs: this.auditLogs(),
      healthMetrics: this.healthMetrics(),
      performanceData: this.performanceData(),
    };
  }
}
