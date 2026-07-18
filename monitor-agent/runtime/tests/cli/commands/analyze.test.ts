/**
 * Unit tests for cli/commands/analyze.ts — AnalyzeCommand.
 *
 * Domain: cli
 *
 * Tests AnalyzeCommand.execute() with existing and missing trace files.
 */

import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import { AnalyzeCommand } from "../../../src/cli/commands/analyze.js";
import { Logger } from "../../../src/shared/logger.js";

describe("AnalyzeCommand", () => {
  const mockCycleRunner = {
    run: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
  };
  const logger = new Logger("error");

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("reports error when trace file does not exist", async () => {
    const command = new AnalyzeCommand(mockCycleRunner as never, logger);
    // Should not throw
    await expect(
      command.execute({ agente: "/path/to/analyzer", trace: "/nonexistent/trace.json" }),
    ).resolves.toBeUndefined();
    expect(mockCycleRunner.run).not.toHaveBeenCalled();
  });

  it("reads trace and calls cycle runner when file exists", async () => {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const testDir = "/tmp/opencode/analyze-test";
    const testTrace = path.join(testDir, "trace.json");

    fs.mkdirSync(testDir, { recursive: true });
    fs.writeFileSync(testTrace, JSON.stringify({
      trace_id: "abc123",
      agente: "test",
      tipo_agente: "task_based",
      entrada: "test input",
      tempo_total_segundos: 1,
      tokens_consumidos: { prompt: 0, completion: 0, total: 0 },
      etapas: [],
    }), "utf-8");

    const command = new AnalyzeCommand(mockCycleRunner as never, logger);
    await command.execute({ agente: "/path/to/analyzer", trace: testTrace });

    expect(mockCycleRunner.run).toHaveBeenCalled();

    fs.unlinkSync(testTrace);
    fs.rmdirSync(testDir);
  });
});
