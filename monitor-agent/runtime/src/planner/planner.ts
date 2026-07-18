/**
 * Planner — decides the next action via LLM or mock fallback.
 *
 * Domain: planner
 *
 * Combines perception and system prompt to call the LLM for structured plan
 * generation. Falls back to a mock planner when no API key is available,
 * cycling through available tools in order. Mirrors the Python runtime's
 * chamar_llm() and planejador_mock() functions.
 *
 * Used by: cycle runner (plan phase), CLI commands.
 */

import { z } from "zod";
import type { Plan } from "./planner.types.js";
import type { AllContracts } from "../contracts/contracts.types.js";
import type { AgentState } from "../core/state.types.js";
import type { TokenUsage } from "../shared/shared.types.js";
import { EMPTY_TOKEN_USAGE } from "../shared/shared.types.js";
import type { StructuredOutputHandler } from "../llm/structured-output.js";
import type { PerceptionBuilder } from "./perception.js";
import type { PromptBuilder } from "./prompt-builder.js";
import type { Logger } from "../shared/logger.js";

/**
 * Zod schema for validating the LLM's structured plan output.
 *
 * Ensures the LLM returns exactly the fields the cycle runner expects.
 * The schema is passed to the AI SDK's generateObject() for constrained
 * decoding on supported providers.
 */
const PlanSchema = z.object({
  proxima_acao: z.enum(["CHAMAR_FERRAMENTA", "FINALIZAR", "PERGUNTAR_USUARIO"]),
  nome_ferramenta: z.string().optional(),
  argumentos_ferramenta: z.record(z.unknown()).optional(),
  criterio_sucesso: z.string(),
  pergunta: z.string().optional(),
});

/**
 * Raw plan type as returned by the LLM before field normalization.
 */
type RawPlan = z.infer<typeof PlanSchema>;

/**
 * Generates plans from the LLM or via mock fallback.
 *
 * Uses StructuredOutputHandler for type-safe LLM calls with Zod validation.
 * When no API key is available, cycles through available tools in order
 * (mock planner) to support testing without credentials.
 *
 * Used by: cycle runner (plan phase).
 */
export class Planner {
  /** Handler for structured LLM output generation with retry logic. */
  private readonly structuredOutput: StructuredOutputHandler;

  /** Builds perception prompts from agent state. */
  private readonly perceptionBuilder: PerceptionBuilder;

  /** Builds the system prompt from contracts. */
  private readonly promptBuilder: PromptBuilder;

  /** Structured logger for debug and info output. */
  private readonly logger: Logger;

  /**
   * @param structuredOutput - Generates validated structured output from the LLM.
   * @param perceptionBuilder - Builds perception prompts from agent state.
   * @param promptBuilder - Builds the system prompt from contracts.
   * @param logger - Structured logger for debug and info output.
   */
  constructor(
    structuredOutput: StructuredOutputHandler,
    perceptionBuilder: PerceptionBuilder,
    promptBuilder: PromptBuilder,
    logger: Logger,
  ) {
    this.structuredOutput = structuredOutput;
    this.perceptionBuilder = perceptionBuilder;
    this.promptBuilder = promptBuilder;
    this.logger = logger;
  }

  /**
   * Generates a plan using the LLM with structured output.
   *
   * Builds the system prompt and perception from contracts and state,
   * then calls the LLM via StructuredOutputHandler for type-safe plan
   * generation. Normalizes the raw LLM output to the Plan interface.
   *
   * @param state - Current agent state for building perception.
   * @param contracts - Full contract set for prompt building.
   * @returns The LLM-generated plan and token usage.
   * @throws Error if the LLM call fails after all retries.
   *
   * Used by: cycle runner (plan phase).
   *
   * Acceptance criteria:
   * - Returns a valid Plan with correct proxima_acao type.
   */
  async plan(
    state: AgentState,
    contracts: AllContracts,
  ): Promise<{ plan: Plan; tokens: TokenUsage }> {
    const perception = this.perceptionBuilder.build(state);
    const systemPrompt = this.promptBuilder.build(contracts);

    this.logger.info("[planejar] Chamando LLM para gerar plano", {
      etapa: state.etapa,
      tipo: state.tipoAgente,
    });

    const result = await this.structuredOutput.generate({
      schema: PlanSchema,
      systemPrompt,
      prompt: perception,
    });

    const plan = Planner.normalizePlan(result.output);

    this.logger.info("[planejar] Plano gerado pela LLM", {
      proxima_acao: plan.proximaAcao,
      ferramenta: plan.nomeFerramenta,
      tokens: result.usage.total,
    });

    return {
      plan,
      tokens: result.usage,
    };
  }

  /**
   * Generates a plan using the mock planner (no LLM).
   *
   * Cycles through available tools in order, returning one tool per call.
   * When all tools have been called, returns FINALIZAR with a summary
   * of the evidence collected from history.
   *
   * For interactive mode, simulates a user question on the first step.
   *
   * @param state - Current agent state for building perception.
   * @param contracts - Full contract set for tool and mode information.
   * @returns The mock plan (always with zero token usage).
   *
   * Used by: cycle runner fallback, testing without API key.
   *
   * Acceptance criteria:
   * - mockPlan() works without API key.
   */
  mockPlan(
    state: AgentState,
    contracts: AllContracts,
  ): { plan: Plan; tokens: TokenUsage } {
    const perception = this.perceptionBuilder.build(state);

    const habilidades = contracts.habilidades.habilidades;
    const nomesFerramentas = habilidades.map((h) => h.nome);

    // Detect agent type from perception
    let tipoAgente = "task_based";
    for (const linha of perception.split("\n")) {
      if (linha.startsWith("Modo: ")) {
        tipoAgente = linha.replace("Modo: ", "").trim();
        break;
      }
    }
    if (tipoAgente === "task_based") {
      tipoAgente = contracts.agente.tipo;
    }

    // Interactive mode: simulate user question on first step
    if (tipoAgente === "interactive" && state.historico.length === 0) {
      return {
        plan: {
          proximaAcao: "PERGUNTAR_USUARIO",
          nomeFerramenta: undefined,
          argumentosFerramenta: undefined,
          criterioSucesso: "obter informacoes iniciais do usuario",
          pergunta: "Qual servico esta com problema e desde quando voce observou o alerta?",
        },
        tokens: { ...EMPTY_TOKEN_USAGE },
      };
    }

    // Find the next unused tool
    for (const nome of nomesFerramentas) {
      if (!(nome in state.chamadasPorFerramenta)) {
        const habilidade = habilidades.find((h) => h.nome === nome);
        const argumentos = Planner.buildMockArguments(habilidade?.entrada ?? {});

        return {
          plan: {
            proximaAcao: "CHAMAR_FERRAMENTA",
            nomeFerramenta: nome,
            argumentosFerramenta: argumentos,
            criterioSucesso: `${nome} executado com sucesso`,
            pergunta: undefined,
          },
          tokens: { ...EMPTY_TOKEN_USAGE },
        };
      }
    }

    // All tools used — finalize with summary
    const evidencias = Planner.extractEvidence(state.historico);
    const resumoPartes: string[] = [];
    for (const [nomeFerramenta, dados] of Object.entries(evidencias)) {
      const campos = Object.entries(dados)
        .filter(([chave]) => !chave.startsWith("_"))
        .map(([chave, valor]) => `${chave}=${valor}`)
        .join(", ");
      resumoPartes.push(`[${nomeFerramenta}] ${campos}`);
    }
    const resumo = resumoPartes.length > 0 ? resumoPartes.join(" | ") : "sem evidencias";

    return {
      plan: {
        proximaAcao: "FINALIZAR",
        nomeFerramenta: undefined,
        argumentosFerramenta: undefined,
        criterioSucesso: `Diagnostico: ${resumo}`,
        pergunta: undefined,
      },
      tokens: { ...EMPTY_TOKEN_USAGE },
    };
  }

  /**
   * Normalizes raw LLM output to the Plan interface.
   *
   * Maps snake_case field names from the LLM to camelCase in the Plan.
   * Ensures optional fields are properly undefined when missing.
   *
   * @param raw - The raw LLM output matching PlanSchema.
   * @returns Normalized Plan object.
   *
   * Used by: plan().
   */
  private static normalizePlan(raw: RawPlan): Plan {
    return {
      proximaAcao: raw.proxima_acao,
      nomeFerramenta: raw.nome_ferramenta,
      argumentosFerramenta: raw.argumentos_ferramenta,
      criterioSucesso: raw.criterio_sucesso,
      pergunta: raw.pergunta,
    };
  }

  /**
   * Builds mock arguments for a tool based on its input schema.
   *
   * Generates placeholder values matching each parameter's type.
   * Used by the mock planner when cycling through tools.
   *
   * @param entrada - The skill's input parameter schema.
   * @returns Mock argument record.
   *
   * Used by: mockPlan().
   */
  private static buildMockArguments(entrada: Record<string, string>): Record<string, unknown> {
    const args: Record<string, unknown> = {};
    for (const [campo, tipo] of Object.entries(entrada)) {
      switch (tipo) {
        case "string":
          args[campo] = `mock_${campo}`;
          break;
        case "int":
          args[campo] = 42;
          break;
        case "float":
          args[campo] = 3.14;
          break;
        case "bool":
          args[campo] = true;
          break;
        case "list":
          args[campo] = [];
          break;
        case "object":
          args[campo] = {};
          break;
        default:
          args[campo] = `mock_${campo}`;
      }
    }
    return args;
  }

  /**
   * Extracts evidence from execution history for the final summary.
   *
   * Collects the most recent action result data for each tool that was
   * called, building a map of tool name -> result data.
   *
   * @param historico - The execution history entries.
   * @returns Map of tool name to their last result data.
   *
   * Used by: mockPlan() for the FINALIZAR summary.
   */
  private static extractEvidence(
    historico: readonly import("../core/cycle.types.js").HistoryEntry[],
  ): Record<string, Record<string, unknown>> {
    const evidencias: Record<string, Record<string, unknown>> = {};

    for (const entry of historico) {
      if (entry.plano.nomeFerramenta && entry.resultadoAcao) {
        evidencias[entry.plano.nomeFerramenta] = entry.resultadoAcao.dados;
      }
    }

    return evidencias;
  }
}
