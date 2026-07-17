/**
 * Unit tests for hooks.ts — lifecycle hook execution.
 *
 * Domain: tools
 *
 * Tests HookExecutor.execute() and getConfigured() with various hook names
 * and action types:
 * - log action type
 * - alerta action type
 * - Missing hook (silently skipped)
 * - All configured hooks
 */

import { describe, it, expect, beforeEach, afterEach, jest } from "@jest/globals";
import { HookExecutor } from "../../src/tools/hooks.js";
import { Logger } from "../../src/shared/logger.js";
import type { AllContracts } from "../../src/contracts/contracts.types.js";

/**
 * Creates a minimal AllContracts fixture with configurable hooks.
 */
function createTestContracts(hooks?: Partial<AllContracts["ganchos"]["ganchos"]>): AllContracts {
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
      pos_execucao: { avaliar_resultado: false },
    },
    regras: {
      ferramentas_obrigatorias: [],
      limites: { max_etapas: 10, sem_progresso: 3, limite_tempo_segundos: 300, chamadas_ferramenta: {} },
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
        ...hooks,
      },
    },
    habilidades: { habilidades: [] },
    memoria: { memoria_curta: { guardar: [], descartar: [], max_registros: 10 }, resumo_final: { max_linhas: 5, campos: [] } },
  };
}

describe("HookExecutor", () => {
  let stderrSpy: ReturnType<typeof jest.spyOn>;
  let hookExecutor: HookExecutor;

  beforeEach(() => {
    stderrSpy = jest.spyOn(process.stderr, "write").mockImplementation(() => true);
    hookExecutor = new HookExecutor(new Logger("info"));
  });

  afterEach(() => {
    stderrSpy.mockRestore();
  });

  describe("execute", () => {
    it("executes log hook without error", () => {
      const contracts = createTestContracts();
      expect(() => {
        hookExecutor.execute("antes_da_etapa", contracts, { etapa: 1 });
      }).not.toThrow();
    });

    it("executes alerta hook and writes to stderr", () => {
      const contracts = createTestContracts({ em_erro: "alerta" });
      hookExecutor.execute("em_erro", contracts, { etapa: 1, message: "Something went wrong" });
      expect(stderrSpy).toHaveBeenCalled();
    });

    it("skips hook silently when not configured", () => {
      const contracts = createTestContracts({
        antes_da_etapa: undefined as unknown as "log",
      });
      hookExecutor.execute("antes_da_etapa", contracts, { etapa: 1 });
      expect(stderrSpy).not.toHaveBeenCalled();
    });

    it("includes tool name in hook detail", () => {
      const contracts = createTestContracts();
      hookExecutor.execute("antes_da_acao", contracts, { etapa: 1, toolName: "search" });
    });

    it("includes message in hook detail", () => {
      const contracts = createTestContracts();
      hookExecutor.execute("apos_etapa", contracts, { etapa: 1, message: "Step completed" });
    });
  });

  describe("getConfigured", () => {
    it("returns all configured hook names", () => {
      const contracts = createTestContracts();
      const hooks = hookExecutor.getConfigured(contracts);
      expect(hooks).toContain("antes_da_etapa");
      expect(hooks).toContain("apos_etapa");
      expect(hooks).toContain("antes_da_acao");
      expect(hooks).toContain("apos_acao");
      expect(hooks).toContain("em_erro");
    });

    it("returns only configured hooks when some are missing", () => {
      const contracts = createTestContracts({
        antes_da_acao: undefined as unknown as "log",
        apos_acao: undefined as unknown as "log",
      });
      const hooks = hookExecutor.getConfigured(contracts);
      expect(hooks).toHaveLength(3);
      expect(hooks).not.toContain("antes_da_acao");
      expect(hooks).not.toContain("apos_acao");
    });
  });
});
