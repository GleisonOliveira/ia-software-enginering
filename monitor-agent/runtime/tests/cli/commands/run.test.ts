/**
 * Unit tests for cli/commands/run.ts — RunCommand.
 *
 * Domain: cli
 *
 * Tests RunCommand.execute() with various options.
 */

import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import { RunCommand } from "../../../src/cli/commands/run.js";
import { Logger } from "../../../src/shared/logger.js";

describe("RunCommand", () => {
  const mockCycleRunner = {
    run: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
  };
  const logger = new Logger("error");
  let command: RunCommand;

  beforeEach(() => {
    jest.clearAllMocks();
    command = new RunCommand(
      mockCycleRunner as never,
      logger,
    );
  });

  it("calls cycleRunner.run with correct config", async () => {
    await command.execute({
      agente: "/path/to/agent",
      entrada: "test input",
    });

    expect(mockCycleRunner.run).toHaveBeenCalledWith({
      agentPath: "/path/to/agent",
      input: "test input",
      mode: undefined,
      event: undefined,
    });
  });

  it("passes mode and event when provided", async () => {
    await command.execute({
      agente: "/path/to/agent",
      entrada: "test input",
      modo: "interactive",
      evento: "deploy_falhou",
    });

    expect(mockCycleRunner.run).toHaveBeenCalledWith({
      agentPath: "/path/to/agent",
      input: "test input",
      mode: "interactive",
      event: "deploy_falhou",
    });
  });
});
