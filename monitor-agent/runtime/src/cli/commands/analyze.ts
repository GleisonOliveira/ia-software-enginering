/**
 * CLI command for analyzing the last execution trace.
 *
 * Domain: cli
 *
 * Reads the last trace.json, extracts a compact summary, and runs it
 * through an analyzer agent to generate a detailed markdown report.
 * The report includes health analysis, performance analysis, compliance
 * checks, and anomaly detection.
 *
 * Used by: CliApp (analyze subcommand).
 */

import fs from "node:fs";
import path from "node:path";
import type { AnalyzeOptions } from "../cli.types.js";
import type { Logger } from "../../shared/logger.js";
import type { CycleRunner } from "../../core/cycle.js";

/**
 * AnalyzeCommand class — generates analysis reports from execution traces.
 *
 * Reads the last trace.json, builds a compact summary for the analyzer agent,
 * runs the analyzer, and generates a markdown report (analise-agente.md).
 *
 * Used by: CliApp (analyze subcommand).
 */
export class AnalyzeCommand {
  /** Cycle runner for executing the analyzer agent. */
  private readonly cycleRunner: CycleRunner;

  /** Structured logger for output. */
  private readonly logger: Logger;

  /**
   * @param cycleRunner - Executes the analyzer agent.
   * @param logger - Structured logger for output.
   */
  constructor(cycleRunner: CycleRunner, logger: Logger) {
    this.cycleRunner = cycleRunner;
    this.logger = logger;
  }

  /**
   * Analyzes the last trace using the specified analyzer agent.
   *
   * @param options - CLI options for the analyze command.
   *
   * Used by: CliApp (analyze subcommand).
   *
   * Acceptance criteria:
   * - AnalyzeCommand generates analise-agente.md.
   */
  async execute(options: AnalyzeOptions): Promise<void> {
    const tracePath = options.trace ?? "trace.json";
    const resolvedPath = path.resolve(tracePath);

    if (!fs.existsSync(resolvedPath)) {
      this.logger.error(`Trace not found: ${resolvedPath}`);
      this.logger.info("Run an agent first to generate the trace.");
      return;
    }

    this.logger.info(`Analyzing trace: ${resolvedPath}`);

    const rawData = fs.readFileSync(resolvedPath, "utf-8");
    const dadosTrace = JSON.parse(rawData) as Record<string, unknown>;

    // Build compact summary for the analyzer
    const entradaTrace = AnalyzeCommand.resumirTrace(dadosTrace);

    this.logger.info(`Running analyzer agent from: ${options.agente}`);

    // Run the analyzer agent with the trace summary as input
    await this.cycleRunner.run({
      agentPath: options.agente,
      input: entradaTrace,
    });

    // Generate markdown report if analysis output exists
    const analisePath = path.resolve("analise.json");
    if (fs.existsSync(analisePath)) {
      const dadosAnalise = JSON.parse(fs.readFileSync(analisePath, "utf-8")) as Record<string, unknown>;
      const relatorio = AnalyzeCommand.gerarRelatorioMd(dadosTrace, dadosAnalise);
      const caminhoMd = path.resolve("analise-agente.md");
      fs.writeFileSync(caminhoMd, relatorio, "utf-8");
      this.logger.info(`Report saved: ${caminhoMd}`);
    }
  }

  /**
   * Extracts a compact trace summary for the analyzer agent input.
   *
   * @param dados - The raw trace data.
   * @returns Compact text summary.
   *
   * Used by: execute().
   */
  private static resumirTrace(dados: Record<string, unknown>): string {
    const linhas: string[] = [];

    linhas.push(`TRACE_ID: ${dados['trace_id'] as string ?? "?"}`);
    linhas.push(`AGENTE: ${dados['agente'] as string ?? "?"}`);
    linhas.push(`TIPO: ${dados['tipo_agente'] as string ?? "?"}`);
    linhas.push(`TEMPO_TOTAL: ${dados['tempo_total_segundos'] as number ?? 0}s`);
    linhas.push(`TOKENS: ${JSON.stringify(dados['tokens_consumidos'] ?? {})}`);

    const etapas = (dados['etapas'] ?? []) as Record<string, unknown>[];
    for (const etapa of etapas) {
      const num = etapa['etapa'] as number ?? "?";
      const plano = (etapa['plano'] ?? {}) as Record<string, unknown>;
      const acao = plano['proxima_acao'] as string ?? "?";
      const ferramenta = plano['nome_ferramenta'] as string ?? "-";
      const resultado = etapa['resultado_acao'] as Record<string, unknown> | undefined;
      const sucesso = resultado?.['sucesso'] as boolean ?? false;
      const avaliacao = (etapa['avaliacao'] ?? {}) as Record<string, unknown>;
      const qualidade = avaliacao['qualidade'] as string ?? "";
      const objetivo = avaliacao['objetivo_alcancado'] as boolean ?? false;
      const motivo = avaliacao['motivo'] as string ?? "";
      const problemas = avaliacao['problemas_saida'] as string[] | undefined;

      linhas.push(
        `ETAPA ${num}: acao=${acao} ferramenta=${ferramenta} sucesso=${sucesso} ` +
        `qualidade=${qualidade} objetivo=${objetivo} motivo=${motivo}` +
        (problemas?.length ? ` problemas=${problemas.join(",")}` : ""),
      );
    }

    // Health metrics
    const hm = dados['health_metrics'] as Record<string, number> | undefined;
    if (hm) {
      linhas.push(
        `HEALTH: taxa_sucesso=${hm['taxa_sucesso_ferramentas'] ?? 0}% ` +
        `circuit_breaker=${hm['circuit_breaker_ativacoes'] ?? 0} ` +
        `payload_falhas=${hm['validacao_payload_falhas'] ?? 0} ` +
        `chamadas_llm=${hm['chamadas_llm'] ?? 0}`,
      );
    }

    // Performance
    const perf = dados['performance_data'] as Record<string, unknown> | undefined;
    if (perf) {
      linhas.push(`PERF_TOKENS: ${JSON.stringify(perf['tokens'] ?? {})}`);
      linhas.push(`PERF_TEMPO_TOTAL_MS: ${perf['tempo_total_ms'] as number ?? 0}`);
      const fases = (perf['fases'] ?? {}) as Record<string, Record<string, number>>;
      for (const [fase, d] of Object.entries(fases)) {
        linhas.push(
          `PERF_FASE ${fase}: media=${d['media_ms']}ms max=${d['max_ms']}ms ` +
          `total=${d['total_ms']}ms contagem=${d['contagem']}`,
        );
      }
    }

    if (dados['resumo']) {
      linhas.push(`RESUMO: ${dados['resumo'] as string}`);
    }

    return linhas.join("\n");
  }

  /**
   * Generates a markdown report from trace and analysis data.
   *
   * @param dadosTrace - Original execution trace data.
   * @param dadosAnalise - Analyzer agent output data.
   * @returns Markdown report string.
   *
   * Used by: execute().
   */
  private static gerarRelatorioMd(
    dadosTrace: Record<string, unknown>,
    dadosAnalise: Record<string, unknown>,
  ): string {
    const md: string[] = [];

    const agente = dadosTrace['agente'] as string ?? "desconhecido";
    const traceId = dadosTrace['trace_id'] as string ?? "?";
    const tipo = dadosTrace['tipo_agente'] as string ?? "?";
    const tempo = dadosTrace['tempo_total_segundos'] as number ?? 0;
    const tokens = (dadosTrace['tokens_consumidos'] ?? {}) as Record<string, number>;

    md.push(`# Analise de Execucao: ${agente}`);
    md.push("");
    md.push(`- **Trace ID:** ${traceId}`);
    md.push(`- **Tipo:** ${tipo}`);
    md.push(`- **Tempo total:** ${tempo}s`);
    md.push(`- **Tokens:** ${tokens['total'] ?? 0} (prompt=${tokens['prompt'] ?? 0}, completion=${tokens['completion'] ?? 0})`);
    md.push("");

    // Execution pipeline table
    const etapasTrace = (dadosTrace['etapas'] ?? []) as Record<string, unknown>[];
    md.push("## Pipeline Executado");
    md.push("");
    md.push("| Etapa | Acao | Ferramenta | Sucesso | Qualidade |");
    md.push("|-------|------|------------|---------|-----------|");

    for (const et of etapasTrace) {
      const num = et['etapa'] as number ?? "?";
      const plano = (et['plano'] ?? {}) as Record<string, unknown>;
      const acao = plano['proxima_acao'] as string ?? "-";
      const ferr = (plano['nome_ferramenta'] as string) ?? "-";
      const res = et['resultado_acao'] as Record<string, unknown> | undefined;
      const suc = res?.['sucesso'] ? "ok" : "falha";
      const qual = ((et['avaliacao'] as Record<string, unknown>)?.['qualidade'] as string) ?? "-";
      md.push(`| ${num} | ${acao} | ${ferr} | ${suc} | ${qual} |`);
    }
    md.push("");

    // Health section
    const saude = ((dadosAnalise['etapas'] as Record<string, unknown>[] | undefined)?.find(
      (e) => (e['plano'] as Record<string, unknown>)?.['nome_ferramenta'] === "analisar_saude",
    )?.['resultado_acao'] as Record<string, Record<string, unknown>> | undefined)?.['dados'];

    const hm = (dadosTrace['health_metrics'] ?? {}) as Record<string, number>;
    md.push("## Saude");
    md.push("");
    md.push(`- **Taxa de sucesso:** ${saude?.['taxa_sucesso'] as string ?? hm['taxa_sucesso_ferramentas'] ?? "?"}%`);
    md.push(`- **Circuit breaker:** ${saude?.['circuit_breaker_ativacoes'] as number ?? hm['circuit_breaker_ativacoes'] ?? 0} ativacoes`);
    md.push("");

    // Performance section
    const perf = (dadosTrace['performance_data'] ?? {}) as Record<string, unknown>;
    md.push("## Performance");
    md.push("");
    md.push(`- **Tempo total:** ${perf['tempo_total_ms'] as number ?? 0}ms`);
    md.push(`- **Tokens usados:** ${tokens['total'] ?? 0}`);
    md.push("");

    md.push("### Detalhamento por Fase");
    md.push("");
    md.push("| Fase | Media | Max | Total | Chamadas |");
    md.push("|------|-------|-----|-------|----------|");
    const fases = (perf['fases'] ?? {}) as Record<string, Record<string, number>>;
    for (const [fase, d] of Object.entries(fases)) {
      md.push(`| ${fase} | ${d['media_ms']}ms | ${d['max_ms']}ms | ${d['total_ms']}ms | ${d['contagem']}x |`);
    }
    md.push("");

    // Summary
    if (dadosTrace['resumo']) {
      md.push("## Resumo");
      md.push("");
      md.push(dadosTrace['resumo'] as string);
      md.push("");
    }

    return md.join("\n");
  }
}
