/**
 * Unit tests for prompt-builder.ts — PromptBuilder.
 *
 * Domain: planner
 *
 * Tests PromptBuilder.build() with various contract configurations:
 * - Basic agent with tools
 * - Agent with no tools
 * - Interactive mode instructions
 * - Goal-oriented mode instructions
 * - Autonomous mode instructions
 * - Agent with planner rules and policies
 */

import { describe, it, expect } from "@jest/globals";
import { PromptBuilder } from "../../src/planner/prompt-builder.js";
import type { AllContracts } from "../../src/contracts/contracts.types.js";

/**
 * Creates a minimal AllContracts fixture for testing.
 */
function createTestContracts(overrides?: Partial<AllContracts>): AllContracts {
  return {
    agente: {
      nome: "Test Agent",
      descricao: "A test agent",
      tipo: "task_based",
      objetivo: "Test objective",
      contrato_saida: { formato: "json", campos_obrigatorios: [], exemplo: {} },
    },
    ciclo: { objetivo: "Test objective", ciclo: { max_etapas: 10 }, condicoes_parada: [] },
    planejador: {
      formato_saida: { proxima_acao: "CHAMAR_FERRAMENTA", criterio_sucesso: "string" },
      regras: ["Use each tool at most once"],
    },
    caixa_ferramentas: { ferramentas: [] },
    executor: {
      execucao: { validar_entrada: false, tentar_novamente_em_falha: false },
      pos_execucao: { avaliar_resultado: false },
    },
    regras: {
      ferramentas_obrigatorias: [],
      limites: { max_etapas: 10, sem_progresso: 3, limite_tempo_segundos: 300, chamadas_ferramenta: {} },
      acoes_sensiveis: [],
      politicas: ["Always be safe"],
    },
    ganchos: { ganchos: { antes_da_etapa: "log", apos_etapa: "log", antes_da_acao: "log", apos_acao: "log", em_erro: "alerta" } },
    habilidades: {
      habilidades: [
        {
          nome: "web_search",
          descricao: "Search the web",
          entrada: { query: "string" },
          saida: { resultado: "string" },
        },
      ],
    },
    memoria: { memoria_curta: { guardar: [], descartar: [], max_registros: 10 }, resumo_final: { max_linhas: 5, campos: [] } },
    ...overrides,
  };
}

describe("PromptBuilder", () => {
  const builder = new PromptBuilder();

  it("builds a prompt with agent identity", () => {
    const contracts = createTestContracts();
    const prompt = builder.build(contracts);

    expect(prompt).toContain("Agente: Test Agent - A test agent");
    expect(prompt).toContain("Tipo: task_based");
    expect(prompt).toContain("Objetivo: Test objective");
  });

  it("includes tool descriptions", () => {
    const contracts = createTestContracts();
    const prompt = builder.build(contracts);

    expect(prompt).toContain("web_search: Search the web");
    expect(prompt).toContain("entrada: {query: string}");
    expect(prompt).toContain("saida: {resultado: string}");
  });

  it("shows 'nenhuma ferramenta disponivel' when no tools", () => {
    const contracts = createTestContracts({
      habilidades: { habilidades: [] },
    });
    const prompt = builder.build(contracts);

    expect(prompt).toContain("nenhuma ferramenta disponivel");
  });

  it("includes response format specification", () => {
    const contracts = createTestContracts();
    const prompt = builder.build(contracts);

    expect(prompt).toContain("CHAMAR_FERRAMENTA");
    expect(prompt).toContain("FINALIZAR");
    expect(prompt).toContain("PERGUNTAR_USUARIO");
  });

  it("includes planner rules", () => {
    const contracts = createTestContracts();
    const prompt = builder.build(contracts);

    expect(prompt).toContain("Use each tool at most once");
  });

  it("includes agent policies", () => {
    const contracts = createTestContracts();
    const prompt = builder.build(contracts);

    expect(prompt).toContain("Always be safe");
  });

  it("includes interactive mode instructions", () => {
    const contracts = createTestContracts({
      agente: { ...createTestContracts().agente, tipo: "interactive" },
    });
    const prompt = builder.build(contracts);

    expect(prompt).toContain("MODO INTERACTIVE");
    expect(prompt).toContain("PERGUNTAR_USUARIO");
  });

  it("includes goal_oriented mode instructions", () => {
    const contracts = createTestContracts({
      agente: { ...createTestContracts().agente, tipo: "goal_oriented" },
    });
    const prompt = builder.build(contracts);

    expect(prompt).toContain("MODO GOAL-ORIENTED");
    expect(prompt).toContain("sub-objetivos");
  });

  it("includes autonomous mode instructions", () => {
    const contracts = createTestContracts({
      agente: { ...createTestContracts().agente, tipo: "autonomous" },
    });
    const prompt = builder.build(contracts);

    expect(prompt).toContain("MODO AUTONOMOUS");
    expect(prompt).toContain("evento trigger");
  });

  it("does not include mode instructions for task_based", () => {
    const contracts = createTestContracts();
    const prompt = builder.build(contracts);

    expect(prompt).not.toContain("MODO INTERACTIVE");
    expect(prompt).not.toContain("MODO GOAL-ORIENTED");
    expect(prompt).not.toContain("MODO AUTONOMOUS");
  });

  it("includes cycle steps", () => {
    const contracts = createTestContracts();
    const prompt = builder.build(contracts);

    expect(prompt).toContain("perceber -> planejar -> agir -> avaliar");
  });
});
