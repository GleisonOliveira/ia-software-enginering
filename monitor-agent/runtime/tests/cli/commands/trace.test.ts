/**
 * Unit tests for cli/commands/trace.ts — TraceCommand.
 *
 * Domain: cli
 *
 * Tests TraceCommand.execute() with existing and missing trace files.
 */

import { describe, it, expect, beforeEach, afterEach } from "@jest/globals";
import fs from "node:fs";
import path from "node:path";
import { TraceCommand } from "../../../src/cli/commands/trace.js";
import { Logger } from "../../../src/shared/logger.js";

describe("TraceCommand", () => {
  const logger = new Logger("error");
  const testTraceDir = path.resolve("/tmp/opencode/trace-test");
  const testTracePath = path.join(testTraceDir, "trace.json");

  beforeEach(() => {
    fs.mkdirSync(testTraceDir, { recursive: true });
  });

  afterEach(() => {
    if (fs.existsSync(testTracePath)) fs.unlinkSync(testTracePath);
    if (fs.existsSync(testTraceDir)) fs.rmdirSync(testTraceDir);
  });

  it("displays error when trace file does not exist", () => {
    const command = new TraceCommand(logger);
    // Should not throw
    expect(() => command.execute({ arquivo: "/nonexistent/trace.json" })).not.toThrow();
  });

  it("reads and displays trace data when file exists", () => {
    const traceData = {
      trace_id: "abc123",
      tipo_agente: "task_based",
      entrada: "test input",
      tempo_total_segundos: 5.5,
      tokens_consumidos: { prompt: 100, completion: 50, total: 150 },
      etapas: [
        {
          etapa: 1,
          plano: { proxima_acao: "CHAMAR_FERRAMENTA", nome_ferramenta: "web_search" },
          resultado_acao: { sucesso: true, dados: { resultado: "found" } },
          avaliacao: { objetivo_alcancado: false, qualidade: "completa" },
        },
      ],
    };

    fs.writeFileSync(testTracePath, JSON.stringify(traceData), "utf-8");

    const command = new TraceCommand(logger);
    expect(() => command.execute({ arquivo: testTracePath })).not.toThrow();
  });
});
