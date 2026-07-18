/**
 * CLI command for displaying the last execution trace.
 *
 * Domain: cli
 *
 * Reads trace.json from the project root (or a specified file) and
 * displays a formatted summary of the execution history, including
 * step-by-step plans, tool results, and evaluations.
 *
 * Used by: CliApp (trace subcommand).
 */

import fs from "node:fs";
import path from "node:path";
import type { TraceOptions } from "../cli.types.js";
import type { Logger } from "../../shared/logger.js";

/**
 * Default path to the trace.json file (project root).
 */
const DEFAULT_TRACE_PATH = "trace.json";

/**
 * TraceCommand class — displays the last execution trace.
 *
 * Reads and formats trace data from trace.json, showing step-by-step
 * execution history, health metrics, performance data, and the final summary.
 *
 * Used by: CliApp (trace subcommand).
 */
export class TraceCommand {
  /** Structured logger for output. */
  private readonly logger: Logger;

  /**
   * @param logger - Structured logger for output.
   */
  constructor(logger: Logger) {
    this.logger = logger;
  }

  /**
   * Displays the trace from the specified or default trace file.
   *
   * @param options - CLI options for the trace command.
   *
   * Used by: CliApp (trace subcommand).
   *
   * Acceptance criteria:
   * - TraceCommand displays the last trace.json.
   */
  execute(options: TraceOptions): void {
    const tracePath = options.arquivo ?? DEFAULT_TRACE_PATH;
    const resolvedPath = path.resolve(tracePath);

    if (!fs.existsSync(resolvedPath)) {
      this.logger.error(`Trace not found: ${resolvedPath}`);
      this.logger.info("Run an agent first to generate the trace.");
      return;
    }

    const rawData = fs.readFileSync(resolvedPath, "utf-8");
    const dados = JSON.parse(rawData) as Record<string, unknown>;

    // Support both old format (array) and new format (object with metadata)
    const historico = Array.isArray(dados) ? dados : (dados['etapas'] as Record<string, unknown>[] ?? []);
    const metadados = Array.isArray(dados) ? {} : dados;

    this.logger.info("\n" + "=".repeat(60));
    this.logger.info("RASTREAMENTO - ultima execucao");

    if (metadados['trace_id']) {
      this.logger.info(`  Trace ID: ${metadados['trace_id'] as string}`);
    }
    if (metadados['tipo_agente']) {
      this.logger.info(`  Tipo: ${metadados['tipo_agente'] as string}`);
    }
    if (metadados['entrada']) {
      this.logger.info(`  Entrada: ${metadados['entrada'] as string}`);
    }
    if (metadados['tempo_total_segundos']) {
      this.logger.info(`  Tempo: ${metadados['tempo_total_segundos'] as number}s`);
    }
    if (metadados['tokens_consumidos']) {
      const tokens = metadados['tokens_consumidos'] as Record<string, number>;
      this.logger.info(`  Tokens: ${tokens['total'] ?? 0} (prompt=${tokens['prompt'] ?? 0}, completion=${tokens['completion'] ?? 0})`);
    }

    this.logger.info("=".repeat(60) + "\n");

    for (const registro of historico) {
      const etapa = registro['etapa'] as number;
      const plano = (registro['plano'] ?? {}) as Record<string, unknown>;
      const resultado = registro['resultado_acao'] as Record<string, unknown> | undefined;
      const avaliacao = (registro['avaliacao'] ?? {}) as Record<string, unknown>;

      this.logger.info(`Etapa ${etapa}`);
      this.logger.info(`  plano     : ${plano['proxima_acao'] as string ?? "?"} -> ${plano['nome_ferramenta'] as string ?? "-"}`);
      this.logger.info(`  criterio  : ${plano['criterio_sucesso'] as string ?? "-"}`);

      if (resultado) {
        const situacao = resultado['sucesso'] ? "ok" : "falha";
        const dadosOuErro = resultado['dados'] ?? resultado['erro'] ?? "";
        const dadosStr = typeof dadosOuErro === "string" ? dadosOuErro : JSON.stringify(dadosOuErro);
        this.logger.info(`  acao      : ${situacao} - ${dadosStr.substring(0, 80)}`);
      }

      const qualidade = avaliacao['qualidade'] as string ?? "";
      this.logger.info(`  avaliacao : objetivo_alcancado=${avaliacao['objetivo_alcancado'] as boolean}${qualidade ? ` qualidade=${qualidade}` : ""}`);
      this.logger.info("");
    }

    // Display health metrics if available
    if (metadados['health_metrics']) {
      const hm = metadados['health_metrics'] as Record<string, number>;
      this.logger.info("--- Health Metrics ---");
      this.logger.info(`  Taxa sucesso: ${hm['taxa_sucesso_ferramentas'] ?? 0}%`);
      this.logger.info(`  Circuit breaker: ${hm['circuit_breaker_ativacoes'] ?? 0}`);
      this.logger.info(`  Payload falhas: ${hm['validacao_payload_falhas'] ?? 0}`);
      this.logger.info("");
    }

    // Display performance data if available
    if (metadados['performance_data']) {
      const perf = metadados['performance_data'] as Record<string, unknown>;
      this.logger.info("--- Performance ---");
      this.logger.info(`  Tokens: ${JSON.stringify(perf['tokens'] ?? {})}`);
      this.logger.info(`  Chamadas LLM: ${perf['chamadas_llm'] ?? 0}`);
      const fases = (perf['fases'] ?? {}) as Record<string, Record<string, number>>;
      for (const [nomeFase, dadosFase] of Object.entries(fases)) {
        this.logger.info(`  ${nomeFase}: media=${dadosFase['media_ms']}ms max=${dadosFase['max_ms']}ms`);
      }
      this.logger.info("");
    }

    // Display summary if available
    if (metadados['resumo']) {
      this.logger.info("--- Resumo ---");
      this.logger.info(metadados['resumo'] as string);
      this.logger.info("");
    }
  }
}
