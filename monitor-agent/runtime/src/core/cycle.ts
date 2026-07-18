/**
 * Main execution cycle — orchestrates perceive→plan→act→evaluate loop.
 *
 * Domain: core
 *
 * The CycleRunner is the central orchestrator that ties all domains together:
 * perception builder, planner, circuit breaker, executor, evaluator, telemetry,
 * and hooks. It manages the main loop, enforces resource limits, handles
 * stagnation detection, and persists the execution trace.
 *
 * Mirrors the Python runtime's rodar() function in ciclo.py.
 *
 * Used by: CLI run command, replay command, analyze command.
 */

import fs from "node:fs";
import path from "node:path";
import type { CycleConfig, HistoryEntry, Trace } from "./cycle.types.js";
import type { AgentState } from "./state.types.js";
import type { Plan } from "../planner/planner.types.js";
import type { AllContracts } from "../contracts/contracts.types.js";
import type { ActionResult, TokenUsage } from "../shared/shared.types.js";
import { EMPTY_TOKEN_USAGE } from "../shared/shared.types.js";
import type { ContractLoader } from "../contracts/loader.js";
import type { StateManager } from "./state.js";
import type { PerceptionBuilder } from "../planner/perception.js";
import type { Planner } from "../planner/planner.js";
import type { CircuitBreaker } from "../executor/circuit-breaker.js";
import type { ToolExecutor } from "../executor/executor.js";
import type { Evaluator } from "../executor/evaluator.js";
import type { ToolRegistry } from "../tools/tool-registry.js";
import type { ToolBuilder } from "../tools/tool-builder.js";
import type { HookExecutor } from "../tools/hooks.js";
import { Telemetry } from "../telemetry/telemetry.js";
import type { Logger } from "../shared/logger.js";

/**
 * Mutable state wrapper allowing internal state mutations.
 *
 * The CycleRunner needs to update state fields during the loop.
 * This type allows mutation while the public AgentState stays immutable.
 */
interface MutableState {
  etapa: number;
  chamadasFerramenta: number;
  chamadasPorFerramenta: Record<string, number>;
  tokensConsumidos: TokenUsage;
  historico: HistoryEntry[];
  concluido: boolean;
  resultado: string;
  etapasSemProgresso: number;
  ultimaFerramenta: string | undefined;
}

/**
 * Orchestrates the full perceive→plan→act→evaluate execution cycle.
 *
 * Ties together all runtime domains: loads contracts, builds tools,
 * creates initial state, and runs the main loop with circuit breaker,
 * payload validation, stagnation detection, and telemetry.
 *
 * Used by: CLI run, replay, and analyze commands.
 */
export class CycleRunner {
  /** Loads agent contracts from .md files. */
  private readonly contractLoader: ContractLoader;

  /** Creates initial agent state from contracts. */
  private readonly stateManager: StateManager;

  /** Builds perception prompts from agent state. */
  private readonly perceptionBuilder: PerceptionBuilder;

  /** Generates plans via LLM or mock fallback. */
  private readonly planner: Planner;

  /** Validates LLM plans against available tools. */
  private readonly circuitBreaker: CircuitBreaker;

  /** Executes tools by name with validation and retry. */
  private readonly toolExecutor: ToolExecutor;

  /** Evaluates step outcomes against success criteria. */
  private readonly evaluator: Evaluator;

  /** Registry of executable tools keyed by name. */
  private readonly toolRegistry: ToolRegistry;

  /** Builds executable tool functions from skill contracts. */
  private readonly toolBuilder: ToolBuilder;

  /** Executes lifecycle hooks defined in contracts. */
  private readonly hookExecutor: HookExecutor;

  /** Structured logger for all runtime output. */
  private readonly logger: Logger;

  /**
   * Creates a CycleRunner with all dependencies injected.
   *
   * @param contractLoader - Loads agent contracts.
   * @param stateManager - Creates initial state.
   * @param perceptionBuilder - Builds perception prompts.
   * @param planner - Generates plans via LLM or mock.
   * @param circuitBreaker - Validates plans against tools.
   * @param toolExecutor - Executes tools by name.
   * @param evaluator - Evaluates step outcomes.
   * @param toolRegistry - Registry of executable tools.
   * @param toolBuilder - Builds tool functions from contracts.
   * @param hookExecutor - Executes lifecycle hooks.
   * @param logger - Structured logger for output.
   */
  constructor(
    contractLoader: ContractLoader,
    stateManager: StateManager,
    perceptionBuilder: PerceptionBuilder,
    planner: Planner,
    circuitBreaker: CircuitBreaker,
    toolExecutor: ToolExecutor,
    evaluator: Evaluator,
    toolRegistry: ToolRegistry,
    toolBuilder: ToolBuilder,
    hookExecutor: HookExecutor,
    logger: Logger,
  ) {
    this.contractLoader = contractLoader;
    this.stateManager = stateManager;
    this.perceptionBuilder = perceptionBuilder;
    this.planner = planner;
    this.circuitBreaker = circuitBreaker;
    this.toolExecutor = toolExecutor;
    this.evaluator = evaluator;
    this.toolRegistry = toolRegistry;
    this.toolBuilder = toolBuilder;
    this.hookExecutor = hookExecutor;
    this.logger = logger;
  }

  /**
   * Runs the full perceive→plan→act→evaluate cycle.
   *
   * Loads contracts, builds tools, creates state, and runs the main loop.
   * The loop continues until the objective is achieved, a limit is hit,
   * or the maximum number of steps is reached.
   *
   * @param config - Execution configuration (agent path, input, mode, event).
   *
   * Used by: CLI run command, replay command.
   *
   * Acceptance criteria:
   * - CycleRunner.run() executes the full perceive→plan→act→evaluate cycle.
   * - Time, token, and stagnation limits interrupt the cycle correctly.
   * - trace.json is saved with all telemetry data.
   * - KPI panel is displayed after each step.
   */
  async run(config: CycleConfig): Promise<void> {
    const contracts = this.contractLoader.loadAllContracts(config.agentPath);
    const state = this.stateManager.createState(contracts, config.input, config.mode, config.event);
    const mutableState = CycleRunner.toMutable(state);

    // Build tools from contracts
    this.toolRegistry.clear();
    const skillEntries = contracts.habilidades.habilidades.map((skill) =>
      this.toolBuilder.buildEntry(skill),
    );
    this.toolRegistry.registerAll(skillEntries);

    // Initialize telemetry
    const tel = new Telemetry(
      path.basename(config.agentPath),
      state.tipoAgente,
    );
    tel.registrar("inicio", {
      entrada: state.entrada,
      objetivo: state.objetivo,
      maxEtapas: state.limits.maxEtapas,
      maxTokens: state.limits.maxTokens,
    });

    // Display execution header
    this.printHeader(config, state, tel);

    const inicio = performance.now();
    const nomesFerramentasDisponiveis = new Set(this.toolRegistry.getNames());

    // Main loop
    while (!mutableState.concluido && mutableState.etapa < state.limits.maxEtapas) {
      mutableState.etapa += 1;

      // Hook: before step
      this.hookExecutor.execute("antes_da_etapa", contracts, { etapa: mutableState.etapa });
      this.logger.info(`--- Etapa ${mutableState.etapa} ---`);

      // Check time limit
      if (this.verificarTempo(inicio, state.limits.limiteTempoSegundos)) {
        this.logger.info(`  [regras] limite de tempo excedido (${state.limits.limiteTempoSegundos}s)`);
        tel.registrar("limite_tempo_excedido", { segundos: state.limits.limiteTempoSegundos });
        mutableState.concluido = true;
        mutableState.resultado = "encerrado por limite de tempo";
        break;
      }

      // Check token limit
      if (this.verificarLimiteTokens(mutableState, state.limits.maxTokens)) {
        this.logger.info(`  [regras] limite de tokens excedido (${mutableState.tokensConsumidos.total}/${state.limits.maxTokens})`);
        tel.registrar("limite_tokens_excedido", { ...mutableState.tokensConsumidos });
        mutableState.concluido = true;
        mutableState.resultado = `encerrado por limite de tokens (${mutableState.tokensConsumidos.total})`;
        break;
      }

      // PHASE: PERCEBER
      const marcadorPerceber = tel.iniciarFase("perceber", mutableState.etapa);
      const stateSnapshot = CycleRunner.toImmutable(mutableState, state);
      const percepcao = this.perceptionBuilder.build(stateSnapshot);
      tel.finalizarFase(marcadorPerceber);
      this.logger.info(`  [perceber] contexto montado (${marcadorPerceber.duracaoMs}ms)`);

      // PHASE: PLANEJAR
      const marcadorPlanejar = tel.iniciarFase("planejar", mutableState.etapa);
      let plano: Plan;
      let usoTokensPlano: TokenUsage;

      try {
        const resultado = await this.planner.plan(stateSnapshot, contracts);
        plano = resultado.plan;
        usoTokensPlano = resultado.tokens;
      } catch {
        // Fallback to mock planner
        this.logger.warn("  [planejar] LLM call failed, using mock planner");
        const resultado = this.planner.mockPlan(stateSnapshot, contracts);
        plano = resultado.plan;
        usoTokensPlano = resultado.tokens;
      }

      tel.finalizarFase(marcadorPlanejar);
      CycleRunner.acumularTokens(mutableState, usoTokensPlano);
      tel.registrarTokens(usoTokensPlano);

      this.logger.info(`  [planejar] proxima_acao=${plano.proximaAcao} ferramenta=${plano.nomeFerramenta} (${marcadorPlanejar.duracaoMs}ms, tokens=${usoTokensPlano.total})`);

      // Circuit breaker validation
      const validacao = this.circuitBreaker.validate(plano, contracts);
      if (!validacao.valido) {
        tel.registrarCircuitBreaker(validacao.erros.join("; "));
        this.logger.info(`  [circuit_breaker] resposta da LLM rejeitada: ${validacao.erros.join("; ")}`);

        // Auto-correction attempts
        const corrigido = this.circuitBreaker.autoCorrect(plano, contracts);
        if (corrigido !== plano) {
          plano = corrigido;
          this.logger.info(`  [circuit_breaker] auto-correcao aplicada`);
        } else {
          // Try fallback to next unused tool
          const fallback = this.findFallbackTool(contracts, mutableState, nomesFerramentasDisponiveis);
          if (fallback) {
            plano = fallback;
            this.logger.info(`  [circuit_breaker] redirecionando para fallback: ${fallback.nomeFerramenta}`);
          } else {
            mutableState.concluido = true;
            mutableState.resultado = `encerrado por circuit breaker: ${validacao.erros.join("; ")}`;
            break;
          }
        }
      }

      tel.registrar("plano_gerado", {
        proxima_acao: plano.proximaAcao,
        nome_ferramenta: plano.nomeFerramenta,
        criterio_sucesso: plano.criterioSucesso,
      });

      // Handle PERGUNTAR_USUARIO
      if (plano.proximaAcao === "PERGUNTAR_USUARIO") {
        const pergunta = plano.pergunta ?? "Preciso de mais informacoes.";
        this.logger.info(`\n  [interactive] ${pergunta}`);

        mutableState.historico.push({
          etapa: mutableState.etapa,
          percepcao,
          plano,
          resultadoAcao: { sucesso: true, dados: { resposta_usuario: "(modo nao-interativo)" }, erro: "", _tokens: EMPTY_TOKEN_USAGE, _entrada: {} },
          avaliacao: { objetivoAlcancado: false, motivo: "aguardando resposta do usuario", qualidade: undefined, problemasSaida: [] },
        });

        this.hookExecutor.execute("apos_etapa", contracts, { etapa: mutableState.etapa });
        continue;
      }

      // Check mandatory tools before FINALIZAR
      if (plano.proximaAcao === "FINALIZAR") {
        const faltantes = contracts.regras.ferramentas_obrigatorias.filter(
          (nome) => !(nome in mutableState.chamadasPorFerramenta),
        );
        if (faltantes.length > 0) {
          this.logger.info(`  [regras] ferramentas obrigatorias pendentes: ${faltantes.join(", ")}`);
          const habilidade = contracts.habilidades.habilidades.find((h) => h.nome === faltantes[0]);
          plano = {
            proximaAcao: "CHAMAR_FERRAMENTA",
            nomeFerramenta: faltantes[0],
            argumentosFerramenta: CycleRunner.buildMockArgs(habilidade?.entrada ?? {}),
            criterioSucesso: `${faltantes[0]} obrigatorio antes de finalizar`,
            pergunta: undefined,
          };
          this.logger.info(`  [regras] redirecionando para: ${faltantes[0]}`);
        }
      }

      // PHASE: AGIR
      let resultadoAcao: ActionResult | undefined;
      if (plano.proximaAcao === "CHAMAR_FERRAMENTA" && plano.nomeFerramenta) {
        const nomeFerramenta = plano.nomeFerramenta;

        // Check total tool call limit
        if (mutableState.chamadasFerramenta >= state.limits.maxChamadasFerramenta) {
          this.logger.info(`  [regras] limite total de chamadas de ferramenta atingido (${state.limits.maxChamadasFerramenta})`);
          mutableState.concluido = true;
          mutableState.resultado = "encerrado por limite total de chamadas de ferramenta";
          break;
        }

        // Check per-tool call limit
        const chamadasDestaferramenta = mutableState.chamadasPorFerramenta[nomeFerramenta] ?? 0;
        const limiteDestaferramenta = state.limits.limitesPorFerramenta[nomeFerramenta];
        if (limiteDestaferramenta !== undefined && chamadasDestaferramenta >= limiteDestaferramenta) {
          this.logger.info(`  [regras] limite de ${nomeFerramenta} atingido (${limiteDestaferramenta})`);
          mutableState.concluido = true;
          mutableState.resultado = `encerrado por limite de ${nomeFerramenta}`;
          break;
        }

        // Check stagnation
        if (CycleRunner.verificarSemProgresso(mutableState, nomeFerramenta, state.limits.semProgresso)) {
          this.logger.info(`  [regras] sem progresso detectado - ${mutableState.etapasSemProgresso} chamadas consecutivas a '${nomeFerramenta}'`);
          mutableState.concluido = true;
          mutableState.resultado = `encerrado por estagnacao (ferramenta repetida: ${nomeFerramenta})`;
          break;
        }

        // Payload validation
        const marcadorValidacao = tel.iniciarFase("validar_payload", mutableState.etapa);
        tel.finalizarFase(marcadorValidacao);

        // Execute tool
        const marcadorAgir = tel.iniciarFase("agir", mutableState.etapa);
        this.hookExecutor.execute("antes_da_acao", contracts, { etapa: mutableState.etapa, toolName: nomeFerramenta });

        const resultado = await this.toolExecutor.execute(
          nomeFerramenta,
          (plano.argumentosFerramenta ?? {}) as Record<string, unknown>,
          this.toolRegistry.toToolMap(),
          contracts,
        );

        tel.finalizarFase(marcadorAgir);
        resultadoAcao = resultado.resultado;

        // Update state
        mutableState.chamadasFerramenta += 1;
        mutableState.chamadasPorFerramenta[nomeFerramenta] = chamadasDestaferramenta + 1;

        // Accumulate tool tokens
        if (resultado.tokensUsados.total > 0) {
          CycleRunner.acumularTokens(mutableState, resultado.tokensUsados);
          tel.registrarTokens(resultado.tokensUsados);
        }

        const sucesso = resultado.resultado.sucesso;
        tel.registrarResultadoFerramenta(sucesso);
        tel.registrar("ferramenta_executada", {
          ferramenta: nomeFerramenta,
          sucesso,
          duracaoMs: marcadorAgir.duracaoMs,
          tokens: resultado.tokensUsados.total,
        });

        this.hookExecutor.execute("apos_acao", contracts, { etapa: mutableState.etapa, toolName: nomeFerramenta });

        if (!sucesso) {
          this.hookExecutor.execute("em_erro", contracts, { etapa: mutableState.etapa, message: resultado.resultado.erro });
        }

        this.logger.info(`  [agir] resultado=${JSON.stringify(resultado.resultado.dados).substring(0, 100)} (${marcadorAgir.duracaoMs}ms)`);
      }

      // PHASE: AVALIAR
      const marcadorAvaliar = tel.iniciarFase("avaliar", mutableState.etapa);
      const avaliacao = this.evaluator.evaluate(plano, resultadoAcao, contracts);
      tel.finalizarFase(marcadorAvaliar);

      this.logger.info(`  [avaliar] objetivo_alcancado=${avaliacao.objetivoAlcancado} - ${avaliacao.motivo} (${marcadorAvaliar.duracaoMs}ms)`);

      // Update history
      mutableState.historico.push({
        etapa: mutableState.etapa,
        percepcao,
        plano,
        resultadoAcao,
        avaliacao,
      });

      if (avaliacao.objetivoAlcancado) {
        mutableState.concluido = true;
        mutableState.resultado = avaliacao.motivo;
      }

      // Hook: after step
      this.hookExecutor.execute("apos_etapa", contracts, { etapa: mutableState.etapa });

      // Display KPI panel
      this.exibirKpis(mutableState, state, tel, inicio, contracts);
    }

    // Finalization
    tel.registrar("finalizado", {
      etapas: mutableState.etapa,
      resultado: mutableState.resultado || "max_etapas_excedido",
      tokensTotal: mutableState.tokensConsumidos.total,
    });

    const tempoTotal = Math.round((performance.now() - inicio) / 1000 * 100) / 100;
    const resumo = CycleRunner.gerarResumoFinal(mutableState, state, contracts);

    // Display final summary
    this.printSummary(mutableState, state, tel, tempoTotal, resumo);

    // Save trace
    const trace = CycleRunner.buildTrace(mutableState, state, tel, tempoTotal, resumo, config);
    const caminhoTrace = config.output ?? "trace.json";
    fs.writeFileSync(caminhoTrace, JSON.stringify(trace, null, 2), "utf-8");
    this.logger.info(`  Rastreamento salvo: ${caminhoTrace}`);
  }

  // --- Private helper methods ---

  private static toMutable(immutable: AgentState): MutableState {
    return {
      etapa: immutable.etapa,
      chamadasFerramenta: immutable.chamadasFerramenta,
      chamadasPorFerramenta: { ...immutable.chamadasPorFerramenta },
      tokensConsumidos: { ...immutable.tokensConsumidos },
      historico: [...immutable.historico],
      concluido: immutable.concluido,
      resultado: immutable.resultado,
      etapasSemProgresso: immutable.etapasSemProgresso,
      ultimaFerramenta: immutable.ultimaFerramenta,
    };
  }

  private static toImmutable(mutable: MutableState, base: AgentState): AgentState {
    return {
      ...base,
      etapa: mutable.etapa,
      chamadasFerramenta: mutable.chamadasFerramenta,
      chamadasPorFerramenta: { ...mutable.chamadasPorFerramenta },
      tokensConsumidos: { ...mutable.tokensConsumidos },
      historico: [...mutable.historico],
      concluido: mutable.concluido,
      resultado: mutable.resultado,
      etapasSemProgresso: mutable.etapasSemProgresso,
      ultimaFerramenta: mutable.ultimaFerramenta,
    };
  }

  private static acumularTokens(state: MutableState, uso: TokenUsage): void {
    state.tokensConsumidos = {
      prompt: state.tokensConsumidos.prompt + uso.prompt,
      completion: state.tokensConsumidos.completion + uso.completion,
      total: state.tokensConsumidos.total + uso.total,
    };
  }

  private verificarTempo(inicio: number, limiteSegundos: number): boolean {
    return (performance.now() - inicio) / 1000 >= limiteSegundos;
  }

  private verificarLimiteTokens(state: MutableState, maxTokens: number): boolean {
    return state.tokensConsumidos.total >= maxTokens;
  }

  private static verificarSemProgresso(
    state: MutableState,
    nomeFerramenta: string,
    limiteSemProgresso: number,
  ): boolean {
    if (nomeFerramenta === state.ultimaFerramenta) {
      state.etapasSemProgresso += 1;
    } else {
      state.etapasSemProgresso = 0;
    }
    state.ultimaFerramenta = nomeFerramenta;
    return state.etapasSemProgresso >= limiteSemProgresso;
  }

  private findFallbackTool(
    contracts: AllContracts,
    state: MutableState,
    disponiveis: Set<string>,
  ): Plan | undefined {
    const habilidades = contracts.habilidades.habilidades;
    const fallback = habilidades.find(
      (h) => disponiveis.has(h.nome) && !(h.nome in state.chamadasPorFerramenta),
    );

    if (fallback) {
      return {
        proximaAcao: "CHAMAR_FERRAMENTA",
        nomeFerramenta: fallback.nome,
        argumentosFerramenta: CycleRunner.buildMockArgs(fallback.entrada),
        criterioSucesso: `fallback apos circuit breaker: ${fallback.nome}`,
        pergunta: undefined,
      };
    }

    return undefined;
  }

  private static buildMockArgs(entrada: Record<string, string>): Record<string, unknown> {
    const args: Record<string, unknown> = {};
    for (const [campo, tipo] of Object.entries(entrada)) {
      switch (tipo) {
        case "string": args[campo] = `mock_${campo}`; break;
        case "int": args[campo] = 42; break;
        case "float": args[campo] = 3.14; break;
        case "bool": args[campo] = true; break;
        case "list": args[campo] = []; break;
        case "object": args[campo] = {}; break;
        default: args[campo] = `mock_${campo}`;
      }
    }
    return args;
  }

  private exibirKpis(
    mutableState: MutableState,
    state: AgentState,
    tel: Telemetry,
    inicio: number,
    contracts: AllContracts,
  ): void {
    const maxEtapas = state.limits.maxEtapas;
    const maxChamadas = state.limits.maxChamadasFerramenta;
    const maxTokens = state.limits.maxTokens;
    const limiteTempo = state.limits.limiteTempoSegundos;

    const tempoDecorrido = Math.round((performance.now() - inicio) / 1000 * 10) / 10;

    // Token bar
    const pctTokens = maxTokens > 0 ? mutableState.tokensConsumidos.total / maxTokens : 0;
    const blocosCheios = Math.floor(pctTokens * 10);
    const barra = "\u2593".repeat(blocosCheios) + "\u2591".repeat(10 - blocosCheios);
    const pctStr = `${(pctTokens * 100).toFixed(1)}%`;

    // Tools status
    const habilidades = contracts.habilidades.habilidades;
    const obrigatorias = new Set(contracts.regras.ferramentas_obrigatorias);
    const partesFerramentas = habilidades.map((h) => {
      if (h.nome in mutableState.chamadasPorFerramenta) return `\u2713${h.nome}`;
      if (obrigatorias.has(h.nome)) return `!${h.nome}`;
      return `\u25cb${h.nome}`;
    });

    // Quality counts
    let ok = 0, parcial = 0, falha = 0;
    for (const h of mutableState.historico) {
      const q = h.avaliacao.qualidade;
      if (q === "completa") ok++;
      else if (q === "parcial") parcial++;
      else if (q === "falha") falha++;
    }

    const cb = tel.healthMetrics().circuitBreakerAtivacoes;
    const pv = tel.healthMetrics().validacaoPayloadFalhas;

    const lat = tel.kpisEtapa(mutableState.etapa);
    const partesLat = Object.entries(lat).map(([fase, ms]) => `${fase}=${Math.round(ms)}ms`);
    const textoLat = partesLat.length > 0 ? partesLat.join("  ") : "-";

    const largura = 58;
    this.logger.info(`\n  \u250c\u2500 KPIs ${"_".repeat(largura - 8)}\u2510`);
    this.logger.info(`  \u2502 Progresso:  ${mutableState.etapa}/${maxEtapas} etapas    ${mutableState.chamadasFerramenta}/${maxChamadas} chamadas    ${tempoDecorrido}s/${limiteTempo}s`);
    this.logger.info(`  \u2502 Tokens:     ${mutableState.tokensConsumidos.total}/${maxTokens} (${pctStr})  ${barra}`);
    this.logger.info(`  \u2502 Ferramentas: ${partesFerramentas.join(" ")}`);
    this.logger.info(`  \u2502 Qualidade:  ${ok}/${ok + parcial + falha} ok   ${parcial} parcial   ${falha} falha`);
    this.logger.info(`  \u2502 Alertas:    ${cb} circuit_breaker   ${pv} payload_invalido`);
    this.logger.info(`  \u2502 Latencia:   ${textoLat}`);
    this.logger.info(`  \u2514${"_".repeat(largura)}\u2518`);
  }

  private printHeader(config: CycleConfig, state: AgentState, tel: Telemetry): void {
    this.logger.info(`\n${"=".repeat(60)}`);
    this.logger.info(`  Agente: ${path.basename(config.agentPath)}`);
    this.logger.info(`  Trace ID: ${tel.traceId}`);
    this.logger.info(`  Tipo: ${state.tipoAgente}`);
    this.logger.info(`  Objetivo: ${state.objetivo}`);
    this.logger.info(`  Entrada: ${state.entrada}`);
    if (state.evento) {
      this.logger.info(`  Evento: ${state.evento}`);
    }
    this.logger.info(`  Max etapas: ${state.limits.maxEtapas}`);
    this.logger.info(`  Limite tempo: ${state.limits.limiteTempoSegundos}s`);
    this.logger.info(`  Limite tokens: ${state.limits.maxTokens}`);
    this.logger.info(`  Ferramentas: ${this.toolRegistry.getNames().join(", ")}`);
    this.logger.info(`${"=".repeat(60)}\n`);
  }

  private printSummary(
    mutableState: MutableState,
    _state: AgentState,
    tel: Telemetry,
    tempoTotal: number,
    resumo: string,
  ): void {
    this.logger.info(`\n${"=".repeat(60)}`);
    this.logger.info(`  Trace ID: ${tel.traceId}`);
    this.logger.info(`  Finalizado em ${mutableState.etapa} etapas (${tempoTotal}s)`);
    this.logger.info(`  Chamadas de ferramenta: ${mutableState.chamadasFerramenta}`);
    this.logger.info(`  Tokens consumidos: ${mutableState.tokensConsumidos.total} (prompt=${mutableState.tokensConsumidos.prompt}, completion=${mutableState.tokensConsumidos.completion})`);
    this.logger.info(`  Resultado: ${mutableState.resultado || "max_etapas_excedido"}`);

    // Health metrics
    const metricas = tel.healthMetrics();
    this.logger.info(`\n  --- Health Metrics ---`);
    this.logger.info(`  Taxa sucesso ferramentas: ${metricas.taxaSucessoFerramentas}%`);
    this.logger.info(`  Circuit breaker ativacoes: ${metricas.circuitBreakerAtivacoes}`);
    this.logger.info(`  Validacao payload falhas: ${metricas.validacaoPayloadFalhas}`);
    this.logger.info(`  Chamadas LLM: ${metricas.chamadasLlm}`);

    // Performance data
    const perf = tel.performanceData();
    this.logger.info(`\n  --- Performance por Fase ---`);
    for (const [nomeFase, dadosFase] of Object.entries(perf.fases)) {
      this.logger.info(`  ${nomeFase}: media=${dadosFase.mediaMs}ms max=${dadosFase.maxMs}ms total=${dadosFase.totalMs}ms (${dadosFase.contagem}x)`);
    }

    this.logger.info(`\n  --- Resumo ---`);
    for (const linha of resumo.split("\n")) {
      this.logger.info(`  ${linha}`);
    }
    this.logger.info(`${"=".repeat(60)}\n`);
  }

  private static gerarResumoFinal(
    mutableState: MutableState,
    state: AgentState,
    contracts: AllContracts,
  ): string {
    const maxLinhas = contracts.memoria.resumo_final.max_linhas;
    const ferramentasChamadas = Object.keys(mutableState.chamadasPorFerramenta);
    const linhas = [
      `Objetivo: ${state.objetivo}`,
      `Etapas executadas: ${mutableState.etapa}`,
      `Ferramentas chamadas: ${ferramentasChamadas.length > 0 ? ferramentasChamadas.join(", ") : "nenhuma"}`,
      `Resultado: ${mutableState.resultado || "max_etapas_excedido"}`,
      `Tipo: ${state.tipoAgente}`,
    ];
    return linhas.slice(0, maxLinhas).join("\n");
  }

  private static buildTrace(
    mutableState: MutableState,
    state: AgentState,
    tel: Telemetry,
    tempoTotal: number,
    resumo: string,
    config: CycleConfig,
  ): Trace {
    const summary = tel.resumoCompleto();

    return {
      traceId: tel.traceId,
      tipoAgente: state.tipoAgente,
      entrada: state.entrada,
      evento: state.evento,
      tempoTotalSegundos: tempoTotal,
      tokensConsumidos: { ...mutableState.tokensConsumidos },
      etapas: [...mutableState.historico],
      resumo,
      agente: path.basename(config.agentPath),
      telemetryStream: summary.telemetryStream,
      auditLogs: summary.auditLogs,
      healthMetrics: summary.healthMetrics,
      performanceData: summary.performanceData,
    };
  }
}
