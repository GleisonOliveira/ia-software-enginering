/**
 * CLI application entry point wrapping Commander.js.
 *
 * Domain: cli
 *
 * Registers all subcommands (run, validate, trace, analyze, replay) with
 * Commander.js and dispatches to the appropriate command handler. Serves
 * as the main entry point for the `monitor-runtime` binary.
 *
 * Used by: package.json bin field, cycle runner entry point.
 */

import { Command } from "commander";
import type { RunCommand } from "./commands/run.js";
import type { ValidateCommand } from "./commands/validate.js";
import type { TraceCommand } from "./commands/trace.js";
import type { AnalyzeCommand } from "./commands/analyze.js";
import type { ReplayCommand } from "./commands/replay.js";

/**
 * CliApp class — wraps Commander.js and registers all subcommands.
 *
 * Receives all command handlers via dependency injection, avoiding
 * module-level singletons. Each handler is a fully formed class with
 * its own dependencies injected.
 *
 * Used by: package.json bin field (monitor-runtime).
 */
export class CliApp {
  /** Commander.js program instance. */
  private readonly program: Command;

  /**
   * Creates the CLI application with all subcommands registered.
   *
   * @param runCommand - Handler for the "run" subcommand.
   * @param validateCommand - Handler for the "validate" subcommand.
   * @param traceCommand - Handler for the "trace" subcommand.
   * @param analyzeCommand - Handler for the "analyze" subcommand.
   * @param replayCommand - Handler for the "replay" subcommand.
   */
  constructor(
    runCommand: RunCommand,
    validateCommand: ValidateCommand,
    traceCommand: TraceCommand,
    analyzeCommand: AnalyzeCommand,
    replayCommand: ReplayCommand,
  ) {
    this.program = new Command();

    this.program
      .name("monitor-runtime")
      .description("Runtime do Agente - AI agent execution runtime")
      .version("1.0.0");

    // Register "run" subcommand
    this.program
      .command("run")
      .description("Execute an agent")
      .requiredOption("--agente <path>", "Path to the agent directory")
      .requiredOption("--entrada <text>", "Agent input text (e.g., alerta de latencia)")
      .option("--modo <mode>", "Operation mode (task_based, interactive, goal_oriented, autonomous)")
      .option("--evento <event>", "Trigger event for autonomous mode")
      .action(async (opts) => {
        await runCommand.execute({
          agente: opts.agente,
          entrada: opts.entrada,
          modo: opts.modo,
          evento: opts.evento,
        });
      });

    // Register "validate" subcommand
    this.program
      .command("validate")
      .description("Validate agent contracts")
      .requiredOption("--agente <path>", "Path to the agent directory")
      .action((opts) => {
        validateCommand.execute({ agente: opts.agente });
      });

    // Register "trace" subcommand
    this.program
      .command("trace")
      .description("Display the last execution trace")
      .option("--arquivo <path>", "Path to a specific trace file")
      .action((opts) => {
        traceCommand.execute({ arquivo: opts.arquivo });
      });

    // Register "analyze" subcommand
    this.program
      .command("analyze")
      .description("Analyze the last execution trace using an analyzer agent")
      .requiredOption("--agente <path>", "Path to the trace-analyzer agent")
      .option("--trace <path>", "Path to a specific trace file")
      .action(async (opts) => {
        await analyzeCommand.execute({
          agente: opts.agente,
          trace: opts.trace,
        });
      });

    // Register "replay" subcommand
    this.program
      .command("replay")
      .description("Re-execute with the same input from the last trace")
      .requiredOption("--agente <path>", "Path to the agent directory")
      .action(async (opts) => {
        await replayCommand.execute({ agente: opts.agente });
      });
  }

  /**
   * Parses command-line arguments and dispatches to the appropriate handler.
   *
   * @param argv - Process arguments (typically process.argv).
   * @returns Promise that resolves when the command completes.
   *
   * Used by: package.json bin entry point.
   */
  async parse(argv: string[]): Promise<void> {
    await this.program.parseAsync(argv);
  }
}
