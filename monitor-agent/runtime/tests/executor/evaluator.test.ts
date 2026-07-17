/**
 * Unit tests for evaluator.ts — post-action evaluation of tool results.
 *
 * Domain: executor
 *
 * Tests Evaluator.evaluate() with various plan/action combinations:
 * - FINALIZAR action (objective achieved)
 * - CHAMAR_FERRAMENTA with success/failure
 * - PERGUNTAR_USUARIO action
 * - Quality ratings (completa, parcial, falha)
 */

import { describe, it, expect } from "@jest/globals";
import { Evaluator } from "../../src/executor/evaluator.js";
import { PayloadValidator } from "../../src/executor/payload-validator.js";
import type { Plan } from "../../src/planner/planner.types.js";
import type { ActionResult } from "../../src/shared/shared.types.js";
import type { AllContracts } from "../../src/contracts/contracts.types.js";

/**
 * Creates a minimal AllContracts fixture for evaluator tests.
 */
function createTestContracts(): AllContracts {
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
    caixa_ferramentas: { ferramentas: [] },
    executor: {
      execucao: { validar_entrada: false, tentar_novamente_em_falha: false },
      pos_execucao: { avaliar_resultado: true },
    },
    regras: {
      ferramentas_obrigatorias: [],
      limites: { max_etapas: 10, sem_progresso: 3, limite_tempo_segundos: 300, chamadas_ferramenta: {} },
      acoes_sensiveis: [],
      politicas: [],
    },
    ganchos: { ganchos: { antes_da_etapa: "log", apos_etapa: "log", antes_da_acao: "log", apos_acao: "log", em_erro: "alerta" } },
    habilidades: {
      habilidades: [
        {
          nome: "test_tool",
          descricao: "Test tool",
          entrada: { query: "string" },
          saida: { result: "string" },
        },
      ],
    },
    memoria: { memoria_curta: { guardar: [], descartar: [], max_registros: 10 }, resumo_final: { max_linhas: 5, campos: [] } },
  };
}

function createPlan(overrides?: Partial<Plan>): Plan {
  return {
    proximaAcao: "CHAMAR_FERRAMENTA",
    nomeFerramenta: "test_tool",
    argumentosFerramenta: { query: "test" },
    criterioSucesso: "Tool executed successfully",
    pergunta: undefined,
    ...overrides,
  };
}

function createActionResult(overrides?: Partial<ActionResult>): ActionResult {
  return {
    sucesso: true,
    dados: { result: "success data" },
    erro: "",
    _tokens: { prompt: 100, completion: 50, total: 150 },
    _entrada: { query: "test" },
    ...overrides,
  };
}

describe("Evaluator", () => {
  const contracts = createTestContracts();
  const evaluator = new Evaluator(new PayloadValidator());

  describe("FINALIZAR action", () => {
    it("returns objetivoAlcancado=true", () => {
      const plan = createPlan({
        proximaAcao: "FINALIZAR",
        nomeFerramenta: undefined,
        argumentosFerramenta: undefined,
      });
      const eval_ = evaluator.evaluate(plan, undefined, contracts);
      expect(eval_.objetivoAlcancado).toBe(true);
      expect(eval_.motivo).toContain("Tool executed successfully");
    });

    it("uses criterioSucesso as motivo", () => {
      const plan = createPlan({
        proximaAcao: "FINALIZAR",
        nomeFerramenta: undefined,
        argumentosFerramenta: undefined,
        criterioSucesso: "All data collected",
      });
      const eval_ = evaluator.evaluate(plan, undefined, contracts);
      expect(eval_.motivo).toBe("All data collected");
    });
  });

  describe("PERGUNTAR_USUARIO action", () => {
    it("returns objetivoAlcancado=false", () => {
      const plan = createPlan({
        proximaAcao: "PERGUNTAR_USUARIO",
        nomeFerramenta: undefined,
        argumentosFerramenta: undefined,
        pergunta: "What do you need?",
      });
      const eval_ = evaluator.evaluate(plan, undefined, contracts);
      expect(eval_.objetivoAlcancado).toBe(false);
      expect(eval_.motivo).toContain("What do you need?");
    });
  });

  describe("CHAMAR_FERRAMENTA with success", () => {
    it("returns qualidade=completa for valid output", () => {
      const plan = createPlan();
      const result = createActionResult();
      const eval_ = evaluator.evaluate(plan, result, contracts);
      expect(eval_.objetivoAlcancado).toBe(false);
      expect(eval_.qualidade).toBe("completa");
      expect(eval_.motivo).toContain("Step OK");
    });

    it("returns qualidade=falha when action result is undefined", () => {
      const plan = createPlan();
      const eval_ = evaluator.evaluate(plan, undefined, contracts);
      expect(eval_.objetivoAlcancado).toBe(false);
      expect(eval_.qualidade).toBe("falha");
    });

    it("returns qualidade=falha when tool execution failed", () => {
      const plan = createPlan();
      const result = createActionResult({ sucesso: false, erro: "Tool crashed" });
      const eval_ = evaluator.evaluate(plan, result, contracts);
      expect(eval_.objetivoAlcancado).toBe(false);
      expect(eval_.qualidade).toBe("falha");
      expect(eval_.motivo).toContain("Tool crashed");
    });
  });

  describe("output validation", () => {
    it("returns problemasSaida when output fields are missing", () => {
      const plan = createPlan();
      const result = createActionResult({ dados: {} });
      const eval_ = evaluator.evaluate(plan, result, contracts);
      expect(eval_.qualidade).toBe("parcial");
      expect(eval_.problemasSaida.length).toBeGreaterThan(0);
    });

    it("returns empty problemasSaida when output validation is disabled", () => {
      const noEvalContracts = { ...contracts, executor: { ...contracts.executor, pos_execucao: { avaliar_resultado: false } } };
      const plan = createPlan();
      const result = createActionResult({ dados: {} });
      const eval_ = evaluator.evaluate(plan, result, noEvalContracts);
      expect(eval_.qualidade).toBe("completa");
      expect(eval_.problemasSaida).toHaveLength(0);
    });
  });
});
