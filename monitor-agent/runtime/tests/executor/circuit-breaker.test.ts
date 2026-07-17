/**
 * Unit tests for circuit-breaker.ts — LLM response validation.
 *
 * Domain: executor
 *
 * Tests CircuitBreaker.validate() and autoCorrect() with various plan types:
 * - Valid plans (CHAMAR_FERRAMENTA, FINALIZAR, PERGUNTAR_USUARIO)
 * - Invalid action types
 * - Missing tool names
 * - Nonexistent tools
 * - Auto-correction of tool name case
 */

import { describe, it, expect } from "@jest/globals";
import { CircuitBreaker } from "../../src/executor/circuit-breaker.js";
import type { Plan } from "../../src/planner/planner.types.js";
import type { AllContracts } from "../../src/contracts/contracts.types.js";

/**
 * Creates a minimal AllContracts fixture with a toolbox for testing.
 */
function createTestContracts(toolNames: string[] = ["web_search"]): AllContracts {
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
    caixa_ferramentas: {
      ferramentas: toolNames.map((nome) => ({ nome, entrada: {} })),
    },
    executor: {
      execucao: { validar_entrada: false, tentar_novamente_em_falha: false },
      pos_execucao: { avaliar_resultado: false },
    },
    regras: {
      ferramentas_obrigatorias: [],
      limites: { max_etapas: 10, sem_progresso: 3, limite_tempo_segundos: 300, chamadas_ferramenta: {} },
      acoes_sensiveis: [],
      politicas: [],
    },
    ganchos: { ganchos: { antes_da_etapa: "log", apos_etapa: "log", antes_da_acao: "log", apos_acao: "log", em_erro: "alerta" } },
    habilidades: { habilidades: [] },
    memoria: { memoria_curta: { guardar: [], descartar: [], max_registros: 10 }, resumo_final: { max_linhas: 5, campos: [] } },
  };
}

function createPlan(overrides?: Partial<Plan>): Plan {
  return {
    proximaAcao: "CHAMAR_FERRAMENTA",
    nomeFerramenta: "web_search",
    argumentosFerramenta: { query: "test" },
    criterioSucesso: "Search completed",
    pergunta: undefined,
    ...overrides,
  };
}

describe("CircuitBreaker", () => {
  const breaker = new CircuitBreaker();

  describe("validate", () => {
    it("returns valid for a correct CHAMAR_FERRAMENTA plan", () => {
      const contracts = createTestContracts();
      const plan = createPlan();
      const result = breaker.validate(plan, contracts);
      expect(result.valido).toBe(true);
      expect(result.erros).toHaveLength(0);
    });

    it("returns valid for FINALIZAR plan", () => {
      const contracts = createTestContracts();
      const plan = createPlan({
        proximaAcao: "FINALIZAR",
        nomeFerramenta: undefined,
        argumentosFerramenta: undefined,
      });
      const result = breaker.validate(plan, contracts);
      expect(result.valido).toBe(true);
    });

    it("returns valid for PERGUNTAR_USUARIO plan with pergunta", () => {
      const contracts = createTestContracts();
      const plan = createPlan({
        proximaAcao: "PERGUNTAR_USUARIO",
        nomeFerramenta: undefined,
        argumentosFerramenta: undefined,
        pergunta: "What do you need?",
      });
      const result = breaker.validate(plan, contracts);
      expect(result.valido).toBe(true);
    });

    it("returns error for invalid action type", () => {
      const contracts = createTestContracts();
      const plan = createPlan({ proximaAcao: "INVALID_ACTION" as Plan["proximaAcao"] });
      const result = breaker.validate(plan, contracts);
      expect(result.valido).toBe(false);
      expect(result.erros[0]).toContain("Invalid action type");
    });

    it("returns error when tool name is missing for CHAMAR_FERRAMENTA", () => {
      const contracts = createTestContracts();
      const plan = createPlan({ nomeFerramenta: undefined });
      const result = breaker.validate(plan, contracts);
      expect(result.valido).toBe(false);
      expect(result.erros.some((e) => e.includes("nomeFerramenta"))).toBe(true);
    });

    it("returns error when tool does not exist in toolbox", () => {
      const contracts = createTestContracts(["web_search"]);
      const plan = createPlan({ nomeFerramenta: "nonexistent_tool" });
      const result = breaker.validate(plan, contracts);
      expect(result.valido).toBe(false);
      expect(result.erros.some((e) => e.includes("not found"))).toBe(true);
    });

    it("returns error when arguments are missing for CHAMAR_FERRAMENTA", () => {
      const contracts = createTestContracts();
      const plan = createPlan({ argumentosFerramenta: undefined });
      const result = breaker.validate(plan, contracts);
      expect(result.valido).toBe(false);
      expect(result.erros.some((e) => e.includes("argumentosFerramenta"))).toBe(true);
    });

    it("returns error when pergunta is missing for PERGUNTAR_USUARIO", () => {
      const contracts = createTestContracts();
      const plan = createPlan({
        proximaAcao: "PERGUNTAR_USUARIO",
        nomeFerramenta: undefined,
        argumentosFerramenta: undefined,
        pergunta: undefined,
      });
      const result = breaker.validate(plan, contracts);
      expect(result.valido).toBe(false);
      expect(result.erros.some((e) => e.includes("pergunta"))).toBe(true);
    });

    it("returns error when criterioSucesso is empty", () => {
      const contracts = createTestContracts();
      const plan = createPlan({ criterioSucesso: "" });
      const result = breaker.validate(plan, contracts);
      expect(result.valido).toBe(false);
      expect(result.erros.some((e) => e.includes("criterioSucesso"))).toBe(true);
    });
  });

  describe("autoCorrect", () => {
    it("returns plan unchanged when tool name matches exactly", () => {
      const contracts = createTestContracts(["web_search"]);
      const plan = createPlan({ nomeFerramenta: "web_search" });
      const corrected = breaker.autoCorrect(plan, contracts);
      expect(corrected.nomeFerramenta).toBe("web_search");
    });

    it("corrects case-insensitive tool name match", () => {
      const contracts = createTestContracts(["web_search"]);
      const plan = createPlan({ nomeFerramenta: "Web_Search" });
      const corrected = breaker.autoCorrect(plan, contracts);
      expect(corrected.nomeFerramenta).toBe("web_search");
    });

    it("returns original plan when no match is found", () => {
      const contracts = createTestContracts(["web_search"]);
      const plan = createPlan({ nomeFerramenta: "completely_different" });
      const corrected = breaker.autoCorrect(plan, contracts);
      expect(corrected.nomeFerramenta).toBe("completely_different");
    });

    it("returns plan unchanged for non-CHAMAR_FERRAMENTA action", () => {
      const contracts = createTestContracts();
      const plan = createPlan({ proximaAcao: "FINALIZAR" });
      const corrected = breaker.autoCorrect(plan, contracts);
      expect(corrected).toBe(plan);
    });
  });
});
