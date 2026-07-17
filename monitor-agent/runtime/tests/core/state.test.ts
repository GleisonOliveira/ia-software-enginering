/**
 * Unit tests for state.ts — AgentState creation from contracts.
 *
 * Domain: core
 *
 * Tests the StateManager class with various contract configurations
 * and CLI override scenarios.
 */

import { StateManager } from "../../src/core/state.js";
import type { AllContracts } from "../../src/contracts/contracts.types.js";

/**
 * Creates a minimal valid AllContracts object for testing.
 *
 * Uses the smallest valid values for each contract field.
 * Caller can override specific fields using spread syntax.
 */
function createTestContracts(overrides?: Partial<AllContracts>): AllContracts {
  return {
    agente: {
      nome: "test-agent",
      descricao: "A test agent",
      tipo: "task_based",
      objetivo: "Test the system",
      contrato_saida: {
        formato: "json",
        campos_obrigatorios: ["result"],
        exemplo: { result: "ok" },
      },
    },
    ciclo: {
      objetivo: "Test the system",
      ciclo: { max_etapas: 5 },
      condicoes_parada: ["objective achieved"],
    },
    planejador: {
      formato_saida: {
        proxima_acao: "CHAMAR_FERRAMENTA",
        criterio_sucesso: "Tool executed",
      },
      regras: ["Always use tools"],
    },
    caixa_ferramentas: {
      ferramentas: [{ nome: "search", entrada: { query: "string" } }],
    },
    executor: {
      execucao: { validar_entrada: true, tentar_novamente_em_falha: false },
      pos_execucao: { avaliar_resultado: true },
    },
    regras: {
      ferramentas_obrigatorias: [],
      limites: {
        max_etapas: 10,
        sem_progresso: 3,
        limite_tempo_segundos: 120,
        chamadas_ferramenta: { total: "ilimitado" },
      },
      acoes_sensiveis: [],
      politicas: [],
    },
    ganchos: {
      ganchos: {
        antes_da_etapa: "log",
        apos_etapa: "log",
        antes_da_acao: "log",
        apos_acao: "log",
        em_erro: "alerta",
      },
    },
    habilidades: {
      habilidades: [
        {
          nome: "search",
          descricao: "Search for info",
          entrada: { query: "string" },
          saida: { results: "string" },
        },
      ],
    },
    memoria: {
      memoria_curta: {
        guardar: ["context"],
        descartar: ["temp"],
        max_registros: 10,
      },
      resumo_final: {
        max_linhas: 20,
        campos: ["result"],
      },
    },
    ...overrides,
  };
}

describe("state", () => {
  let stateManager: StateManager;

  beforeEach(() => {
    stateManager = new StateManager();
  });

  describe("createState", () => {
    it("returns an AgentState with all required fields populated", () => {
      const contracts = createTestContracts();
      const state = stateManager.createState(contracts, "user input here");

      expect(state.objetivo).toBe("Test the system");
      expect(state.entrada).toBe("user input here");
      expect(state.tipoAgente).toBe("task_based");
      expect(state.evento).toBeUndefined();
    });

    it("initializes step tracking to zero", () => {
      const contracts = createTestContracts();
      const state = stateManager.createState(contracts, "input");

      expect(state.etapa).toBe(0);
      expect(state.chamadasFerramenta).toBe(0);
      expect(state.chamadasPorFerramenta).toEqual({});
    });

    it("initializes execution state to empty/false", () => {
      const contracts = createTestContracts();
      const state = stateManager.createState(contracts, "input");

      expect(state.historico).toEqual([]);
      expect(state.concluido).toBe(false);
      expect(state.resultado).toBe("");
      expect(state.etapasSemProgresso).toBe(0);
      expect(state.ultimaFerramenta).toBeUndefined();
    });

    it("initializes token usage to zero", () => {
      const contracts = createTestContracts();
      const state = stateManager.createState(contracts, "input");

      expect(state.tokensConsumidos).toEqual({ prompt: 0, completion: 0, total: 0 });
    });

    it("extracts limits from regras.md contract", () => {
      const contracts = createTestContracts();
      const state = stateManager.createState(contracts, "input");

      expect(state.limits.maxEtapas).toBe(10);
      expect(state.limits.semProgresso).toBe(3);
      expect(state.limits.limiteTempoSegundos).toBe(120);
    });

    it("copies sensitive actions from regras contract", () => {
      const contracts = createTestContracts({
        regras: {
          ...createTestContracts().regras,
          acoes_sensiveis: ["delete_file", "send_email"],
        },
      });
      const state = stateManager.createState(contracts, "input");

      expect(state.acoesSensiveis).toEqual(["delete_file", "send_email"]);
    });

    it("uses CLI mode override for agent type", () => {
      const contracts = createTestContracts();
      const state = stateManager.createState(contracts, "input", "interactive");

      expect(state.tipoAgente).toBe("interactive");
    });

    it("falls back to contract tipo when mode is undefined", () => {
      const contracts = createTestContracts({
        agente: {
          ...createTestContracts().agente,
          tipo: "goal_oriented",
        },
      });
      const state = stateManager.createState(contracts, "input", undefined);

      expect(state.tipoAgente).toBe("goal_oriented");
    });

    it("sets evento from the event parameter", () => {
      const contracts = createTestContracts();
      const state = stateManager.createState(contracts, "input", undefined, "user-clicked-button");

      expect(state.evento).toBe("user-clicked-button");
    });

    it("handles per-tool call limits from regras", () => {
      const contracts = createTestContracts({
        regras: {
          ...createTestContracts().regras,
          limites: {
            ...createTestContracts().regras.limites,
            chamadas_ferramenta: { search: 5, api_call: 3 },
          },
        },
      });
      const state = stateManager.createState(contracts, "input");

      expect(state.limits.limitesPorFerramenta).toEqual({ search: 5, api_call: 3 });
      expect(state.limits.maxChamadasFerramenta).toBe(8);
    });

    it("returns a fresh object on each call (no shared references)", () => {
      const contracts = createTestContracts();
      const state1 = stateManager.createState(contracts, "input1");
      const state2 = stateManager.createState(contracts, "input2");

      expect(state1).not.toBe(state2);
      expect(state1.entrada).toBe("input1");
      expect(state2.entrada).toBe("input2");
      expect(state1.historico).not.toBe(state2.historico);
      expect(state1.chamadasPorFerramenta).not.toBe(state2.chamadasPorFerramenta);
    });
  });
});
