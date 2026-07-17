/**
 * Unit tests for payload-validator.ts — tool input/output validation.
 *
 * Domain: executor
 *
 * Tests PayloadValidator.validate() and validateOutput() with various tool schemas:
 * - Missing required fields
 * - Type mismatches
 * - Valid payloads
 * - Output validation for successful/failed results
 */

import { describe, it, expect } from "@jest/globals";
import { PayloadValidator } from "../../src/executor/payload-validator.js";
import type { AllContracts } from "../../src/contracts/contracts.types.js";

/**
 * Creates a minimal AllContracts fixture for testing.
 */
function createTestContracts(overrides?: {
  skills?: Array<{ nome: string; descricao: string; entrada: Record<string, "string" | "int" | "float" | "bool" | "list" | "object">; saida: Record<string, "string" | "int" | "float" | "bool" | "list" | "object"> }>;
}): AllContracts {
  const skills = overrides?.skills ?? [
    {
      nome: "test_tool",
      descricao: "A test tool",
      entrada: { query: "string" as const, count: "int" as const },
      saida: { result: "string" as const, data: "object" as const },
    },
  ];

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
    executor: { execucao: { validar_entrada: true, tentar_novamente_em_falha: false }, pos_execucao: { avaliar_resultado: true } },
    regras: { ferramentas_obrigatorias: [], limites: { max_etapas: 10, sem_progresso: 3, limite_tempo_segundos: 300, chamadas_ferramenta: {} }, acoes_sensiveis: [], politicas: [] },
    ganchos: { ganchos: { antes_da_etapa: "log", apos_etapa: "log", antes_da_acao: "log", apos_acao: "log", em_erro: "alerta" } },
    habilidades: { habilidades: skills },
    memoria: { memoria_curta: { guardar: [], descartar: [], max_registros: 10 }, resumo_final: { max_linhas: 5, campos: [] } },
  };
}

describe("PayloadValidator", () => {
  const validator = new PayloadValidator();

  describe("validate", () => {
    it("returns empty array for valid args", () => {
      const contracts = createTestContracts();
      const errors = validator.validate("test_tool", { query: "test", count: 5 }, contracts);
      expect(errors).toHaveLength(0);
    });

    it("returns error when tool not found", () => {
      const contracts = createTestContracts();
      const errors = validator.validate("nonexistent_tool", {}, contracts);
      expect(errors).toHaveLength(1);
      expect(errors[0]).toContain("not found");
    });

    it("returns error for missing required field", () => {
      const contracts = createTestContracts();
      const errors = validator.validate("test_tool", { query: "test" }, contracts);
      expect(errors).toHaveLength(1);
      expect(errors[0]).toContain("count");
      expect(errors[0]).toContain("missing");
    });

    it("returns error for type mismatch", () => {
      const contracts = createTestContracts();
      const errors = validator.validate("test_tool", { query: "test", count: "not_a_number" }, contracts);
      expect(errors).toHaveLength(1);
      expect(errors[0]).toContain("count");
      expect(errors[0]).toContain("expected int");
    });

    it("returns multiple errors for multiple issues", () => {
      const contracts = createTestContracts();
      const errors = validator.validate("test_tool", {}, contracts);
      expect(errors).toHaveLength(2);
    });

    it("handles float type correctly", () => {
      const contracts = createTestContracts({
        skills: [{ nome: "float_tool", descricao: "Float tool", entrada: { value: "float" as const }, saida: {} }],
      });
      expect(validator.validate("float_tool", { value: 42 }, contracts)).toHaveLength(0);
      expect(validator.validate("float_tool", { value: 3.14 }, contracts)).toHaveLength(0);
      expect(validator.validate("float_tool", { value: "string" }, contracts)).toHaveLength(1);
    });

    it("handles bool type correctly", () => {
      const contracts = createTestContracts({
        skills: [{ nome: "bool_tool", descricao: "Bool tool", entrada: { flag: "bool" as const }, saida: {} }],
      });
      expect(validator.validate("bool_tool", { flag: true }, contracts)).toHaveLength(0);
      expect(validator.validate("bool_tool", { flag: "yes" }, contracts)).toHaveLength(1);
    });

    it("handles list type correctly", () => {
      const contracts = createTestContracts({
        skills: [{ nome: "list_tool", descricao: "List tool", entrada: { items: "list" as const }, saida: {} }],
      });
      expect(validator.validate("list_tool", { items: [1, 2, 3] }, contracts)).toHaveLength(0);
      expect(validator.validate("list_tool", { items: "not_a_list" }, contracts)).toHaveLength(1);
    });

    it("handles object type correctly", () => {
      const contracts = createTestContracts({
        skills: [{ nome: "obj_tool", descricao: "Object tool", entrada: { data: "object" as const }, saida: {} }],
      });
      expect(validator.validate("obj_tool", { data: { key: "value" } }, contracts)).toHaveLength(0);
      expect(validator.validate("obj_tool", { data: "not_object" }, contracts)).toHaveLength(1);
    });
  });

  describe("validateOutput", () => {
    it("returns empty array for valid output", () => {
      const contracts = createTestContracts();
      const result = validator.validateOutput(
        "test_tool",
        { sucesso: true, dados: { result: "ok", data: { key: "value" } } },
        contracts,
      );
      expect(result).toHaveLength(0);
    });

    it("returns empty array when tool execution failed", () => {
      const contracts = createTestContracts();
      const result = validator.validateOutput(
        "test_tool",
        { sucesso: false, dados: {} },
        contracts,
      );
      expect(result).toHaveLength(0);
    });

    it("returns error for missing output field", () => {
      const contracts = createTestContracts();
      const result = validator.validateOutput(
        "test_tool",
        { sucesso: true, dados: { result: "ok" } },
        contracts,
      );
      expect(result).toHaveLength(1);
      expect(result[0]).toContain("data");
      expect(result[0]).toContain("missing");
    });

    it("returns error for null output field", () => {
      const contracts = createTestContracts();
      const result = validator.validateOutput(
        "test_tool",
        { sucesso: true, dados: { result: "ok", data: null } },
        contracts,
      );
      expect(result).toHaveLength(1);
      expect(result[0]).toContain("null");
    });

    it("returns error for empty string output field", () => {
      const contracts = createTestContracts({
        skills: [{ nome: "text_tool", descricao: "Text tool", entrada: {}, saida: { text: "string" as const } }],
      });
      const result = validator.validateOutput(
        "text_tool",
        { sucesso: true, dados: { text: "  " } },
        contracts,
      );
      expect(result).toHaveLength(1);
      expect(result[0]).toContain("empty string");
    });

    it("returns empty array for tool not in skills", () => {
      const contracts = createTestContracts();
      const result = validator.validateOutput(
        "unknown_tool",
        { sucesso: true, dados: { anything: true } },
        contracts,
      );
      expect(result).toHaveLength(0);
    });
  });
});
