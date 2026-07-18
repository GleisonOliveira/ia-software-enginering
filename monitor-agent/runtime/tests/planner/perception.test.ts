/**
 * Unit tests for perception.ts — PerceptionBuilder.
 *
 * Domain: planner
 *
 * Tests PerceptionBuilder.build() with various agent states:
 * - Basic state with input and mode
 * - State with event context
 * - State with history entries and tool results
 * - State with stagnation warning
 * - State with tools already used
 */

import { describe, it, expect } from "@jest/globals";
import { PerceptionBuilder } from "../../src/planner/perception.js";
import type { AgentState } from "../../src/core/state.types.js";
import type { HistoryEntry } from "../../src/core/cycle.types.js";
import { EMPTY_TOKEN_USAGE } from "../../src/shared/shared.types.js";

/**
 * Creates a minimal AgentState fixture for testing.
 */
function createTestState(overrides?: Partial<AgentState>): AgentState {
  return {
    objetivo: "Test objective",
    entrada: "alerta de latencia",
    tipoAgente: "task_based",
    evento: undefined,
    etapa: 0,
    chamadasFerramenta: 0,
    chamadasPorFerramenta: {},
    limits: {
      maxEtapas: 10,
      maxChamadasFerramenta: 20,
      limitesPorFerramenta: {},
      semProgresso: 3,
      limiteTempoSegundos: 300,
      maxTokens: 50000,
    },
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

describe("PerceptionBuilder", () => {
  const builder = new PerceptionBuilder();

  it("builds perception with basic input and mode", () => {
    const state = createTestState();
    const perception = builder.build(state);

    expect(perception).toContain("Alerta: alerta de latencia");
    expect(perception).toContain("Modo: task_based");
    expect(perception).toContain("Etapas realizadas: 0/10");
    expect(perception).toContain("Chamadas de ferramenta: 0/20");
  });

  it("includes event context when present", () => {
    const state = createTestState({ evento: "deploy_falhou" });
    const perception = builder.build(state);

    expect(perception).toContain("Evento trigger: deploy_falhou");
  });

  it("does not include event when undefined", () => {
    const state = createTestState({ evento: undefined });
    const perception = builder.build(state);

    expect(perception).not.toContain("Evento trigger");
  });

  it("includes history entries with tool results", () => {
    const historyEntry: HistoryEntry = {
      etapa: 1,
      percepcao: "test perception",
      plano: {
        proximaAcao: "CHAMAR_FERRAMENTA",
        nomeFerramenta: "web_search",
        argumentosFerramenta: { query: "test" },
        criterioSucesso: "search done",
        pergunta: undefined,
      },
      resultadoAcao: {
        sucesso: true,
        dados: { resultado: "found data" },
        erro: "",
        _tokens: EMPTY_TOKEN_USAGE,
        _entrada: { query: "test" },
      },
      avaliacao: {
        objetivoAlcancado: false,
        motivo: "continue",
        qualidade: "completa",
        problemasSaida: [],
      },
    };

    const state = createTestState({ historico: [historyEntry], etapa: 1 });
    const perception = builder.build(state);

    expect(perception).toContain("Etapa 1 [web_search]:");
    expect(perception).toContain("resultado");
  });

  it("includes tools already used", () => {
    const state = createTestState({
      chamadasPorFerramenta: { web_search: 2, calculator: 1 },
    });
    const perception = builder.build(state);

    expect(perception).toContain("Ferramentas ja utilizadas: web_search, calculator");
  });

  it("does not include tools line when no tools used", () => {
    const state = createTestState({ chamadasPorFerramenta: {} });
    const perception = builder.build(state);

    expect(perception).not.toContain("Ferramentas ja utilizadas");
  });

  it("includes stagnation warning when etapasSemProgresso > 0", () => {
    const state = createTestState({ etapasSemProgresso: 2 });
    const perception = builder.build(state);

    expect(perception).toContain("ATENCAO: 2 etapas sem progresso detectadas");
  });

  it("does not include stagnation warning when etapasSemProgresso is 0", () => {
    const state = createTestState({ etapasSemProgresso: 0 });
    const perception = builder.build(state);

    expect(perception).not.toContain("ATENCAO");
  });

  it("formats failed action results correctly", () => {
    const historyEntry: HistoryEntry = {
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
        sucesso: false,
        dados: {},
        erro: "connection timeout",
        _tokens: EMPTY_TOKEN_USAGE,
        _entrada: {},
      },
      avaliacao: {
        objetivoAlcancado: false,
        motivo: "failed",
        qualidade: "falha",
        problemasSaida: [],
      },
    };

    const state = createTestState({ historico: [historyEntry], etapa: 1 });
    const perception = builder.build(state);

    expect(perception).toContain("FALHA: connection timeout");
  });
});
