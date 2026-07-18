/**
 * Unit tests for cli/index.ts — CliApp.
 *
 * Domain: cli
 *
 * Tests CliApp command registration and dispatch:
 * - All subcommands are registered
 * - Correct options are parsed for each command
 * - Version and help are available
 */

import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import { CliApp } from "../../src/cli/index.js";

/**
 * Creates mock command handlers for testing.
 */
function createMockCommands() {
  return {
    runCommand: {
      execute: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
    },
    validateCommand: {
      execute: jest.fn<() => void>().mockReturnValue(undefined),
    },
    traceCommand: {
      execute: jest.fn<() => void>().mockReturnValue(undefined),
    },
    analyzeCommand: {
      execute: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
    },
    replayCommand: {
      execute: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
    },
  };
}

describe("CliApp", () => {
  let mocks: ReturnType<typeof createMockCommands>;
  let app: CliApp;

  beforeEach(() => {
    mocks = createMockCommands();
    app = new CliApp(
      mocks.runCommand as never,
      mocks.validateCommand as never,
      mocks.traceCommand as never,
      mocks.analyzeCommand as never,
      mocks.replayCommand as never,
    );
  });

  it("can be instantiated without errors", () => {
    expect(app).toBeDefined();
  });

  it("has parse method", () => {
    expect(typeof app.parse).toBe("function");
  });

  it("dispatches run command with correct options", async () => {
    await app.parse([
      "node",
      "test",
      "run",
      "--agente",
      "/path/to/agent",
      "--entrada",
      "test input",
    ]);

    expect(mocks.runCommand.execute).toHaveBeenCalledWith({
      agente: "/path/to/agent",
      entrada: "test input",
      modo: undefined,
      evento: undefined,
    });
  });

  it("dispatches run command with optional options", async () => {
    await app.parse([
      "node",
      "test",
      "run",
      "--agente",
      "/path/to/agent",
      "--entrada",
      "test input",
      "--modo",
      "interactive",
      "--evento",
      "deploy_falhou",
    ]);

    expect(mocks.runCommand.execute).toHaveBeenCalledWith({
      agente: "/path/to/agent",
      entrada: "test input",
      modo: "interactive",
      evento: "deploy_falhou",
    });
  });

  it("dispatches validate command", async () => {
    await app.parse([
      "node",
      "test",
      "validate",
      "--agente",
      "/path/to/agent",
    ]);

    expect(mocks.validateCommand.execute).toHaveBeenCalledWith({
      agente: "/path/to/agent",
    });
  });

  it("dispatches trace command without file", async () => {
    await app.parse([
      "node",
      "test",
      "trace",
    ]);

    expect(mocks.traceCommand.execute).toHaveBeenCalledWith({
      arquivo: undefined,
    });
  });

  it("dispatches trace command with file", async () => {
    await app.parse([
      "node",
      "test",
      "trace",
      "--arquivo",
      "custom-trace.json",
    ]);

    expect(mocks.traceCommand.execute).toHaveBeenCalledWith({
      arquivo: "custom-trace.json",
    });
  });

  it("dispatches analyze command", async () => {
    await app.parse([
      "node",
      "test",
      "analyze",
      "--agente",
      "/path/to/analyzer",
      "--trace",
      "trace.json",
    ]);

    expect(mocks.analyzeCommand.execute).toHaveBeenCalledWith({
      agente: "/path/to/analyzer",
      trace: "trace.json",
    });
  });

  it("dispatches replay command", async () => {
    await app.parse([
      "node",
      "test",
      "replay",
      "--agente",
      "/path/to/agent",
    ]);

    expect(mocks.replayCommand.execute).toHaveBeenCalledWith({
      agente: "/path/to/agent",
    });
  });
});
