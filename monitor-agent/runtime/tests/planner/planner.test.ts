/**
 * Unit tests for planner.ts — Planner.
 *
 * Domain: planner
 *
 * Tests Planner.mockPlan() with various agent states:
 * - First step with unused tools
 * - All tools used (finalize)
 * - Interactive mode (user question)
 * - History with evidence extraction
 *
 * Note: plan() is not tested here because it requires mocking the
 * StructuredOutputHandler which involves the AI SDK. The mock planner
 * covers the core planning logic without external dependencies.
 */

import { describe, it, expect } from "@jest/globals";
import { Planner } from "../../src/planner/planner.js";
import { PerceptionBuilder } from "../../src/planner/perception.js";
import { PromptBuilder } from "../../src/planner/prompt-builder.js";
import { Logger } from "../../src/shared/logger.js";
import type { AgentState } from "../../src/core/state.types.js";
import type { AllContracts } from "../../src/contracts/contracts.types.js";
import { EMPTY_TOKEN_USAGE } from "../../src/shared/shared.types.js";

/**
 * Creates a minimal AllContracts fixture for testing.
 */
function createTestContracts(toolNames: string[] = ["web_search", "calculator"]): AllContracts {
  return {
    agente: {
      nome: "Test Agent",
      descricao: "Test",
      tipo: "task_based",
      objetivo: "Test",
      contrato_saida: { formato: "json", campos_obrigatorios: [], exemplo: {} },
    },
    ciclo: { objetivo: "Test", ciclo: { max_etapas: 10 }, condicoes_parada: [] },
    planejador: { formato_saida: { proxima_acao: "CHAMAR_FERRAMENTA", criterio_sucesso: "string" }, regras: [] },
    caixa_ferramentas: { ferramentas: toolNames.map((nome) => ({ nome, entrada: {} })) },
    executor: { execucao: { validar_entrada: false, tentar_novamente_em_falha: false }, pos_execucao: { avaliar_resultado: false } },
    regras: {
      ferramentas_obrigatorias: [],
      limites: { max_etapas: 10, sem_progresso: 3, limite_tempo_segundos: 300, chamadas_ferramenta: {} },
      acoes_sensiveis: [],
      politicas: [],
    },
    ganchos: { ganchos: { antes_da_etapa: "log", apos_etapa: "log", antes_da_acao: "log", apos_acao: "log", em_erro: "alerta" } },
    habilidades: {
      habilidades: toolNames.map((nome) => ({
        nome,
        descricao: `${nome} tool`,
        entrada: { query: "string" },
        saida: { resultado: "string" },
      })),
    },
    memoria: { memoria_curta: { guardar: [], descartar: [], max_registros: 10 }, resumo_final: { max_linhas: 5, campos: [] } },
  };
}

function createTestState(overrides?: Partial<AgentState>): AgentState {
  return {
    objetivo: "Test",
    entrada: "test input",
    tipoAgente: "task_based",
    evento: undefined,
    etapa: 0,
    chamadasFerramenta: 0,
    chamadasPorFerramenta: {},
    limits: { maxEtapas: 10, maxChamadasFerramenta: 20, limitesPorFerramenta: {}, semProgresso: 3, limiteTempoSegundos: 300, maxTokens: 50000 },
    tokensConsumidos: { ...EMPTY_TOKEN_USAGE },
    acoesSensiveis: [],
    historico: [],
    concluido: false,
    resultado: "",
    etapasSemProgresso: 0,
    ultimaFerramenta: undefined,
    ...overrides,
  };
}

describe("Planner", () => {
  const logger = new Logger("error");
  const perceptionBuilder = new PerceptionBuilder();
  const promptBuilder = new PromptBuilder();

  // StructuredOutputHandler is not needed for mockPlan tests
  const mockStructuredOutput = {} as never;
  const planner = new Planner(mockStructuredOutput, perceptionBuilder, promptBuilder, logger);

  describe("mockPlan", () => {
    it("returns CHAMAR_FERRAMENTA for first unused tool", () => {
      const state = createTestState();
      const contracts = createTestContracts(["web_search"]);
      const result = planner.mockPlan(state, contracts);

      expect(result.plan.proximaAcao).toBe("CHAMAR_FERRAMENTA");
      expect(result.plan.nomeFerramenta).toBe("web_search");
      expect(result.plan.criterioSucesso).toContain("web_search");
      expect(result.tokens.total).toBe(0);
    });

    it("skips already used tools", () => {
      const state = createTestState({
        chamadasPorFerramenta: { web_search: 1 },
      });
      const contracts = createTestContracts(["web_search", "calculator"]);
      const result = planner.mockPlan(state, contracts);

      expect(result.plan.proximaAcao).toBe("CHAMAR_FERRAMENTA");
      expect(result.plan.nomeFerramenta).toBe("calculator");
    });

    it("returns FINALIZAR when all tools are used", () => {
      const state = createTestState({
        chamadasPorFerramenta: { web_search: 1, calculator: 1 },
      });
      const contracts = createTestContracts(["web_search", "calculator"]);
      const result = planner.mockPlan(state, contracts);

      expect(result.plan.proximaAcao).toBe("FINALIZAR");
      expect(result.plan.criterioSucesso).toContain("Diagnostico");
    });

    it("returns PERGUNTAR_USUARIO in interactive mode with no history", () => {
      const state = createTestState({ tipoAgente: "interactive" });
      const contracts = createTestContracts();
      const result = planner.mockPlan(state, contracts);

      expect(result.plan.proximaAcao).toBe("PERGUNTAR_USUARIO");
      expect(result.plan.pergunta).toContain("servico");
    });

    it("returns zero token usage for mock plans", () => {
      const state = createTestState();
      const contracts = createTestContracts();
      const result = planner.mockPlan(state, contracts);

      expect(result.tokens).toEqual(EMPTY_TOKEN_USAGE);
    });

    it("builds mock arguments based on skill input schema", () => {
      const state = createTestState();
      const contracts = createTestContracts(["web_search"]);
      const result = planner.mockPlan(state, contracts);

      expect(result.plan.argumentosFerramenta).toBeDefined();
      expect(result.plan.argumentosFerramenta).toHaveProperty("query");
    });

    it("includes evidence from history in FINALIZAR summary", () => {
      const state = createTestState({
        chamadasPorFerramenta: { web_search: 1 },
        historico: [
          {
            etapa: 1,
            percepcao: "test",
            plano: {
              proximaAcao: "CHAMAR_FERRAMENTA",
              nomeFerramenta: "web_search",
              argumentosFerramenta: {},
              criterioSucesso: "done",
              pergunta: undefined,
            },
            resultadoAcao: {
              sucesso: true,
              dados: { resultado: "found data" },
              erro: "",
              _tokens: EMPTY_TOKEN_USAGE,
              _entrada: {},
            },
            avaliacao: { objetivoAlcancado: false, motivo: "continue", qualidade: "completa", problemasSaida: [] },
          },
        ],
      });
      const contracts = createTestContracts(["web_search"]);
      const result = planner.mockPlan(state, contracts);

      expect(result.plan.proximaAcao).toBe("FINALIZAR");
      expect(result.plan.criterioSucesso).toContain("web_search");
    });
  });
});
