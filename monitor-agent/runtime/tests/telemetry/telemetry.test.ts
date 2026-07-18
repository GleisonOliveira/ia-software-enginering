/**
 * Unit tests for telemetry.ts — Telemetry collector.
 *
 * Domain: telemetry
 *
 * Tests Telemetry class methods:
 * - Event registration with timestamps
 * - Phase timing (start/end)
 * - Token usage accumulation
 * - Tool success/failure counting
 * - Circuit breaker activation tracking
 * - Health metrics computation
 * - Performance data aggregation
 * - Audit log filtering
 * - KPI per-step latencies
 * - Complete summary generation
 */

import { describe, it, expect, beforeEach } from "@jest/globals";
import { Telemetry } from "../../src/telemetry/telemetry.js";

describe("Telemetry", () => {
  let tel: Telemetry;

  beforeEach(() => {
    tel = new Telemetry("test-agent", "task_based");
  });

  describe("constructor", () => {
    it("generates a unique trace ID", () => {
      expect(tel.traceId).toBeDefined();
      expect(tel.traceId.length).toBe(12);
    });

    it("stores agent name and type", () => {
      expect(tel.agente).toBe("test-agent");
      expect(tel.tipoAgente).toBe("task_based");
    });
  });

  describe("registrar", () => {
    it("records an event with timestamp and trace ID", () => {
      tel.registrar("inicio", { entrada: "test" });
      const stream = tel.telemetryStream();

      expect(stream).toHaveLength(1);
      const event = stream[0];
      expect(event).toBeDefined();
      expect(event!.tipo).toBe("inicio");
      expect(event!.traceId).toBe(tel.traceId);
      expect(event!.dados['entrada']).toBe("test");
      expect(event!.timestamp).toBeDefined();
      expect(event!.elapsedMs).toBeGreaterThanOrEqual(0);
    });

    it("records multiple events", () => {
      tel.registrar("inicio");
      tel.registrar("plano_gerado");
      tel.registrar("finalizado");

      expect(tel.telemetryStream()).toHaveLength(3);
    });
  });

  describe("iniciarFase / finalizarFase", () => {
    it("tracks phase timing correctly", () => {
      const marker = tel.iniciarFase("perceber", 1);
      expect(marker.fase).toBe("perceber");
      expect(marker.etapa).toBe(1);
      expect(marker.duracaoMs).toBeUndefined();

      // Simulate some work
      tel.finalizarFase(marker);

      const stream = tel.telemetryStream();
      const phaseEvent = stream.find((e) => e.tipo === "fase_concluida");
      expect(phaseEvent).toBeDefined();
      expect(phaseEvent?.dados['fase']).toBe("perceber");
    });

    it("records phase in performance data", () => {
      const marker = tel.iniciarFase("planejar", 1);
      tel.finalizarFase(marker);

      const perf = tel.performanceData();
      expect(perf.fases['planejar']).toBeDefined();
      expect(perf.fases['planejar']!.contagem).toBe(1);
    });
  });

  describe("registrarTokens", () => {
    it("accumulates token usage", () => {
      tel.registrarTokens({ prompt: 100, completion: 50, total: 150 });
      tel.registrarTokens({ prompt: 200, completion: 100, total: 300 });

      const perf = tel.performanceData();
      expect(perf.tokens.prompt).toBe(300);
      expect(perf.tokens.completion).toBe(150);
      expect(perf.tokens.total).toBe(450);
      expect(perf.chamadasLlm).toBe(2);
    });
  });

  describe("registrarResultadoFerramenta", () => {
    it("counts successes and failures", () => {
      tel.registrarResultadoFerramenta(true);
      tel.registrarResultadoFerramenta(true);
      tel.registrarResultadoFerramenta(false);

      const health = tel.healthMetrics();
      expect(health.ferramentasSucesso).toBe(2);
      expect(health.ferramentasFalha).toBe(1);
      expect(health.taxaSucessoFerramentas).toBeCloseTo(66.7, 1);
    });

    it("returns 0% success rate when no tools executed", () => {
      const health = tel.healthMetrics();
      expect(health.taxaSucessoFerramentas).toBe(0);
    });
  });

  describe("registrarCircuitBreaker", () => {
    it("increments activation count and records event", () => {
      tel.registrarCircuitBreaker("invalid action");
      tel.registrarCircuitBreaker("missing tool");

      const health = tel.healthMetrics();
      expect(health.circuitBreakerAtivacoes).toBe(2);

      const cbEvents = tel.telemetryStream().filter((e) => e.tipo === "circuit_breaker");
      expect(cbEvents).toHaveLength(2);
    });
  });

  describe("registrarValidacaoPayloadFalha", () => {
    it("increments failure count and records event", () => {
      tel.registrarValidacaoPayloadFalha("web_search", ["missing query"]);

      const health = tel.healthMetrics();
      expect(health.validacaoPayloadFalhas).toBe(1);
    });
  });

  describe("auditLogs", () => {
    it("filters audit-relevant events", () => {
      tel.registrar("inicio");
      tel.registrar("plano_gerado");
      tel.registrar("ferramenta_executada");
      tel.registrar("fase_concluida");
      tel.registrar("finalizado");

      const audit = tel.auditLogs();
      expect(audit).toHaveLength(3);
      expect(audit.map((e) => e.tipo)).toEqual(
        expect.arrayContaining(["plano_gerado", "ferramenta_executada", "finalizado"]),
      );
    });

    it("excludes non-audit events", () => {
      tel.registrar("inicio");
      tel.registrar("fase_concluida");

      const audit = tel.auditLogs();
      expect(audit).toHaveLength(0);
    });
  });

  describe("healthMetrics", () => {
    it("returns complete health metrics", () => {
      tel.registrarTokens({ prompt: 100, completion: 50, total: 150 });
      tel.registrarResultadoFerramenta(true);
      tel.registrarCircuitBreaker("test");
      tel.registrarValidacaoPayloadFalha("tool", ["err"]);

      const health = tel.healthMetrics();
      expect(health.traceId).toBe(tel.traceId);
      expect(health.chamadasLlm).toBe(1);
      expect(health.ferramentasSucesso).toBe(1);
      expect(health.ferramentasFalha).toBe(0);
      expect(health.circuitBreakerAtivacoes).toBe(1);
      expect(health.validacaoPayloadFalhas).toBe(1);
    });
  });

  describe("performanceData", () => {
    it("aggregates phase timing correctly", () => {
      const m1 = tel.iniciarFase("perceber", 1);
      tel.finalizarFase(m1);
      const m2 = tel.iniciarFase("planejar", 1);
      tel.finalizarFase(m2);

      const perf = tel.performanceData();
      expect(perf.fases['perceber']!.contagem).toBe(1);
      expect(perf.fases['planejar']!.contagem).toBe(1);
      expect(perf.tempoTotalMs).toBeGreaterThanOrEqual(0);
    });

    it("calculates max and avg correctly", () => {
      const m1 = tel.iniciarFase("agir", 1);
      tel.finalizarFase(m1);
      const m2 = tel.iniciarFase("agir", 2);
      tel.finalizarFase(m2);

      const perf = tel.performanceData();
      expect(perf.fases['agir']!.contagem).toBe(2);
      expect(perf.fases['agir']!.maxMs).toBeGreaterThanOrEqual(0);
    });
  });

  describe("kpisEtapa", () => {
    it("returns latencies for planejar and agir phases", () => {
      const m1 = tel.iniciarFase("planejar", 1);
      tel.finalizarFase(m1);
      const m2 = tel.iniciarFase("agir", 1);
      tel.finalizarFase(m2);
      tel.iniciarFase("perceber", 2);
      const m3 = tel.iniciarFase("planejar", 2);
      tel.finalizarFase(m3);

      const latencias = tel.kpisEtapa(1);
      expect(latencias['planejar']).toBeDefined();
      expect(latencias['agir']).toBeDefined();
      expect(latencias['perceber']).toBeUndefined();
    });

    it("returns empty object for step with no phases", () => {
      const latencias = tel.kpisEtapa(99);
      expect(Object.keys(latencias)).toHaveLength(0);
    });
  });

  describe("resumoCompleto", () => {
    it("returns complete summary with all fields", () => {
      tel.registrar("inicio");
      tel.registrarTokens({ prompt: 100, completion: 50, total: 150 });

      const summary = tel.resumoCompleto();
      expect(summary.traceId).toBe(tel.traceId);
      expect(summary.agente).toBe("test-agent");
      expect(summary.tipoAgente).toBe("task_based");
      expect(summary.telemetryStream).toHaveLength(1);
      expect(summary.healthMetrics).toBeDefined();
      expect(summary.performanceData).toBeDefined();
      expect(summary.auditLogs).toHaveLength(0);
    });
  });
});
