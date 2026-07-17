/**
 * Unit tests for executor.ts — tool execution with retry logic.
 *
 * Domain: executor
 *
 * Tests ToolExecutor.execute() with various scenarios:
 * - Tool not found
 * - Payload validation failure
 * - Successful tool execution
 * - Retry on failure
 * - All attempts failing
 */

import { describe, it, expect } from "@jest/globals";
import { ToolExecutor } from "../../src/executor/executor.js";
import { PayloadValidator } from "../../src/executor/payload-validator.js";
import { Logger } from "../../src/shared/logger.js";
import type { ToolFunction } from "../../src/tools/tools.types.js";
import type { AllContracts } from "../../src/contracts/contracts.types.js";
import type { ActionResult } from "../../src/shared/shared.types.js";

/**
 * Creates a minimal AllContracts fixture for executor tests.
 */
function createTestContracts(overrides?: {
  validar_entrada?: boolean;
  tentar_novamente_em_falha?: boolean;
  avaliar_resultado?: boolean;
}): AllContracts {
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
      execucao: {
        validar_entrada: overrides?.validar_entrada ?? false,
        tentar_novamente_em_falha: overrides?.tentar_novamente_em_falha ?? false,
      },
      pos_execucao: { avaliar_resultado: overrides?.avaliar_resultado ?? false },
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

/**
 * Creates a mock tool function that returns a successful result.
 */
function createSuccessfulTool(name: string): ToolFunction {
  return async (args: Record<string, unknown>): Promise<ActionResult> => ({
    sucesso: true,
    dados: { result: `${name}_output` },
    erro: "",
    _tokens: { prompt: 10, completion: 5, total: 15 },
    _entrada: args,
  });
}

/**
 * Creates a mock tool function that throws an error.
 */
function createFailingTool(errorMessage: string): ToolFunction {
  return async (): Promise<ActionResult> => {
    throw new Error(errorMessage);
  };
}

describe("ToolExecutor", () => {
  const logger = new Logger("error");
  const executor = new ToolExecutor(new PayloadValidator(), logger);

  describe("execute", () => {
    it("returns error when tool not found", async () => {
      const tools = new Map<string, ToolFunction>();
      const contracts = createTestContracts();
      const result = await executor.execute("nonexistent", {}, tools, contracts);
      expect(result.resultado.sucesso).toBe(false);
      expect(result.resultado.erro).toContain("not found");
    });

    it("executes tool successfully", async () => {
      const tools = new Map<string, ToolFunction>([
        ["search", createSuccessfulTool("search")],
      ]);
      const contracts = createTestContracts();
      const result = await executor.execute("search", { query: "test" }, tools, contracts);
      expect(result.resultado.sucesso).toBe(true);
      expect(result.resultado.dados).toEqual({ result: "search_output" });
      expect(result.tokensUsados.total).toBe(15);
    });

    it("retries on failure when tentar_novamente_em_falha is true", async () => {
      let attempts = 0;
      const flakyTool: ToolFunction = async (args: Record<string, unknown>): Promise<ActionResult> => {
        attempts++;
        if (attempts === 1) {
          throw new Error("Transient error");
        }
        return {
          sucesso: true,
          dados: { result: "recovered" },
          erro: "",
          _tokens: { prompt: 10, completion: 5, total: 15 },
          _entrada: args,
        };
      };

      const tools = new Map<string, ToolFunction>([["flaky", flakyTool]]);
      const contracts = createTestContracts({ tentar_novamente_em_falha: true });
      const result = await executor.execute("flaky", {}, tools, contracts);
      expect(result.resultado.sucesso).toBe(true);
      expect(attempts).toBe(2);
    });

    it("returns failure after all retry attempts exhausted", async () => {
      const tools = new Map<string, ToolFunction>([
        ["always_fails", createFailingTool("Permanent error")],
      ]);
      const contracts = createTestContracts({ tentar_novamente_em_falha: true });
      const result = await executor.execute("always_fails", {}, tools, contracts);
      expect(result.resultado.sucesso).toBe(false);
      expect(result.resultado.erro).toContain("Permanent error");
    });

    it("does not retry when tentar_novamente_em_falha is false", async () => {
      let attempts = 0;
      const countingTool: ToolFunction = async (_args: Record<string, unknown>): Promise<ActionResult> => {
        attempts++;
        throw new Error("Error");
      };

      const tools = new Map<string, ToolFunction>([["counter", countingTool]]);
      const contracts = createTestContracts({ tentar_novamente_em_falha: false });
      await executor.execute("counter", {}, tools, contracts);
      expect(attempts).toBe(1);
    });
  });
});
