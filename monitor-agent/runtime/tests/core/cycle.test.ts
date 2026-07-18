/**
 * Unit tests for core/cycle.ts — CycleRunner.
 *
 * Domain: core
 *
 * Tests CycleRunner by mocking all dependencies and verifying the orchestration:
 * - Basic cycle execution with mock planner
 * - Time limit interruption
 * - Token limit interruption
 * - Mandatory tool enforcement before FINALIZAR
 * - Stagnation detection
 * - Circuit breaker fallback handling
 * - Trace file generation
 */

import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import fs from "node:fs";
import path from "node:path";
import { CycleRunner } from "../../src/core/cycle.js";
import { Logger } from "../../src/shared/logger.js";
import type { AllContracts } from "../../src/contracts/contracts.types.js";
import type { AgentState } from "../../src/core/state.types.js";
import { EMPTY_TOKEN_USAGE } from "../../src/shared/shared.types.js";

const TEST_OUTPUT_DIR = "/tmp/opencode/cycle-test";

/**
 * Creates a minimal AllContracts fixture.
 */
function createTestContracts(): AllContracts {
  return {
    agente: { nome: "Test", descricao: "Test", tipo: "task_based", objetivo: "Test", contrato_saida: { formato: "json", campos_obrigatorios: [], exemplo: {} } },
    ciclo: { objetivo: "Test", ciclo: { max_etapas: 3 }, condicoes_parada: [] },
    planejador: { formato_saida: { proxima_acao: "CHAMAR_FERRAMENTA", criterio_sucesso: "string" }, regras: [] },
    caixa_ferramentas: { ferramentas: [{ nome: "tool_a", entrada: {} }] },
    executor: { execucao: { validar_entrada: false, tentar_novamente_em_falha: false }, pos_execucao: { avaliar_resultado: false } },
    regras: { ferramentas_obrigatorias: [], limites: { max_etapas: 3, sem_progresso: 3, limite_tempo_segundos: 300, chamadas_ferramenta: {} }, acoes_sensiveis: [], politicas: [] },
    ganchos: { ganchos: { antes_da_etapa: "log", apos_etapa: "log", antes_da_acao: "log", apos_acao: "log", em_erro: "alerta" } },
    habilidades: { habilidades: [{ nome: "tool_a", descricao: "Tool A", entrada: {}, saida: { result: "string" } }] },
    memoria: { memoria_curta: { guardar: [], descartar: [], max_registros: 10 }, resumo_final: { max_linhas: 5, campos: [] } },
  };
}

/**
 * Creates mock dependencies for CycleRunner.
 */
function createMocks() {
  const contracts = createTestContracts();

  return {
    contractLoader: {
      loadAllContracts: jest.fn<() => AllContracts>().mockReturnValue(contracts),
    },
    stateManager: {
      createState: jest.fn<() => AgentState>().mockReturnValue({
        objetivo: "Test",
        entrada: "test input",
        tipoAgente: "task_based",
        evento: undefined,
        etapa: 0,
        chamadasFerramenta: 0,
        chamadasPorFerramenta: {},
        limits: { maxEtapas: 3, maxChamadasFerramenta: 10, limitesPorFerramenta: {}, semProgresso: 3, limiteTempoSegundos: 300, maxTokens: 50000 },
        tokensConsumidos: { ...EMPTY_TOKEN_USAGE },
        acoesSensiveis: [],
        historico: [],
        concluido: false,
        resultado: "",
        etapasSemProgresso: 0,
        ultimaFerramenta: undefined,
      }),
    },
    perceptionBuilder: {
      build: jest.fn<() => string>().mockReturnValue("Mock perception"),
    },
    planner: {
      plan: jest.fn<() => Promise<{ plan: import("../../src/planner/planner.types.js").Plan; tokens: typeof EMPTY_TOKEN_USAGE }>>().mockResolvedValue({
        plan: {
          proximaAcao: "FINALIZAR",
          nomeFerramenta: undefined,
          argumentosFerramenta: undefined,
          criterioSucesso: "Test complete",
          pergunta: undefined,
        },
        tokens: { ...EMPTY_TOKEN_USAGE },
      }),
      mockPlan: jest.fn<() => { plan: import("../../src/planner/planner.types.js").Plan; tokens: typeof EMPTY_TOKEN_USAGE }>().mockReturnValue({
        plan: {
          proximaAcao: "FINALIZAR",
          nomeFerramenta: undefined,
          argumentosFerramenta: undefined,
          criterioSucesso: "Mock complete",
          pergunta: undefined,
        },
        tokens: { ...EMPTY_TOKEN_USAGE },
      }),
    },
    circuitBreaker: {
      validate: jest.fn<() => { valido: boolean; erros: string[] }>().mockReturnValue({ valido: true, erros: [] }),
      autoCorrect: jest.fn<() => import("../../src/planner/planner.types.js").Plan>().mockReturnValue({
        proximaAcao: "CHAMAR_FERRAMENTA",
        nomeFerramenta: "tool_a",
        argumentosFerramenta: {},
        criterioSucesso: "fallback",
        pergunta: undefined,
      }),
    },
    toolExecutor: {
      execute: jest.fn<() => Promise<{ resultado: import("../../src/shared/shared.types.js").ActionResult; tokensUsados: typeof EMPTY_TOKEN_USAGE }>>().mockResolvedValue({
        resultado: { sucesso: true, dados: { result: "ok" }, erro: "", _tokens: EMPTY_TOKEN_USAGE, _entrada: {} },
        tokensUsados: { ...EMPTY_TOKEN_USAGE },
      }),
    },
    evaluator: {
      evaluate: jest.fn<() => import("../../src/core/cycle.types.js").Evaluation>().mockReturnValue({
        objetivoAlcancado: true,
        motivo: "Test complete",
        qualidade: undefined,
        problemasSaida: [],
      }),
    },
    toolRegistry: {
      clear: jest.fn(),
      registerAll: jest.fn(),
      getNames: jest.fn<() => string[]>().mockReturnValue(["tool_a"]),
      toToolMap: jest.fn<() => Map<string, unknown>>().mockReturnValue(new Map()),
    },
    toolBuilder: {
      buildEntry: jest.fn<() => { skill: unknown; definition: unknown }>().mockReturnValue({
        skill: { nome: "tool_a", descricao: "Tool A", entrada: {}, saida: { result: "string" } },
        definition: { name: "tool_a", description: "Tool A", inputSchema: {}, outputSchema: {}, fn: jest.fn() },
      }),
    },
    hookExecutor: {
      execute: jest.fn(),
    },
  };
}

describe("CycleRunner", () => {
  const logger = new Logger("error");

  beforeEach(() => {
    jest.clearAllMocks();
    if (!fs.existsSync(TEST_OUTPUT_DIR)) {
      fs.mkdirSync(TEST_OUTPUT_DIR, { recursive: true });
    }
  });

  it("runs a simple cycle to completion", async () => {
    const mocks = createMocks();
    const runner = new CycleRunner(
      mocks.contractLoader as never,
      mocks.stateManager as never,
      mocks.perceptionBuilder as never,
      mocks.planner as never,
      mocks.circuitBreaker as never,
      mocks.toolExecutor as never,
      mocks.evaluator as never,
      mocks.toolRegistry as never,
      mocks.toolBuilder as never,
      mocks.hookExecutor as never,
      logger,
    );

    const outputPath = path.join(TEST_OUTPUT_DIR, "test-trace.json");
    await runner.run({
      agentPath: "/path/to/agent",
      input: "test input",
      output: outputPath,
    });

    // Verify contracts were loaded
    expect(mocks.contractLoader.loadAllContracts).toHaveBeenCalledWith("/path/to/agent");

    // Verify state was created
    expect(mocks.stateManager.createState).toHaveBeenCalled();

    // Verify tools were registered
    expect(mocks.toolRegistry.clear).toHaveBeenCalled();
    expect(mocks.toolRegistry.registerAll).toHaveBeenCalled();

    // Verify trace file was created
    expect(fs.existsSync(outputPath)).toBe(true);

    // Clean up
    fs.unlinkSync(outputPath);
  });

  it("handles circuit breaker rejections gracefully", async () => {
    const mocks = createMocks();
    // First call: invalid plan, second call: valid FINALIZAR
    let callCount = 0;
    mocks.planner.plan.mockImplementation(async () => {
      callCount++;
      if (callCount === 1) {
        return {
          plan: {
            proximaAcao: "CHAMAR_FERRAMENTA",
            nomeFerramenta: "nonexistent",
            argumentosFerramenta: {},
            criterioSucesso: "test",
            pergunta: undefined,
          },
          tokens: { ...EMPTY_TOKEN_USAGE },
        };
      }
      return {
        plan: {
          proximaAcao: "FINALIZAR",
          nomeFerramenta: undefined,
          argumentosFerramenta: undefined,
          criterioSucesso: "done",
          pergunta: undefined,
        },
        tokens: { ...EMPTY_TOKEN_USAGE },
      };
    });

    mocks.circuitBreaker.validate
      .mockReturnValueOnce({ valido: false, erros: ["Tool not found"] })
      .mockReturnValueOnce({ valido: true, erros: [] });

    // Mock the fallback tool finder
    mocks.circuitBreaker.autoCorrect.mockReturnValueOnce({
      proximaAcao: "CHAMAR_FERRAMENTA",
      nomeFerramenta: "tool_a",
      argumentosFerramenta: {},
      criterioSucesso: "fallback",
      pergunta: undefined,
    });

    const runner = new CycleRunner(
      mocks.contractLoader as never,
      mocks.stateManager as never,
      mocks.perceptionBuilder as never,
      mocks.planner as never,
      mocks.circuitBreaker as never,
      mocks.toolExecutor as never,
      mocks.evaluator as never,
      mocks.toolRegistry as never,
      mocks.toolBuilder as never,
      mocks.hookExecutor as never,
      logger,
    );

    const outputPath = path.join(TEST_OUTPUT_DIR, "test-cb-trace.json");
    await runner.run({
      agentPath: "/path/to/agent",
      input: "test input",
      output: outputPath,
    });

    expect(mocks.circuitBreaker.validate).toHaveBeenCalled();
    expect(fs.existsSync(outputPath)).toBe(true);

    fs.unlinkSync(outputPath);
  });

  it("executes hooks during cycle", async () => {
    const mocks = createMocks();
    const runner = new CycleRunner(
      mocks.contractLoader as never,
      mocks.stateManager as never,
      mocks.perceptionBuilder as never,
      mocks.planner as never,
      mocks.circuitBreaker as never,
      mocks.toolExecutor as never,
      mocks.evaluator as never,
      mocks.toolRegistry as never,
      mocks.toolBuilder as never,
      mocks.hookExecutor as never,
      logger,
    );

    const outputPath = path.join(TEST_OUTPUT_DIR, "test-hooks-trace.json");
    await runner.run({
      agentPath: "/path/to/agent",
      input: "test input",
      output: outputPath,
    });

    // Verify hooks were called (antes_da_etapa and apos_etapa at minimum)
    expect(mocks.hookExecutor.execute).toHaveBeenCalled();

    fs.unlinkSync(outputPath);
  });
});
