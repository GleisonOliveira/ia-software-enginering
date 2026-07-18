/**
 * CLI command for replaying the last execution.
 *
 * Domain: cli
 *
 * Reads the last trace.json to extract the original input, agent type,
 * and event context, then re-runs the agent with the same parameters.
 * Useful for debugging and verifying fixes without manual re-entry.
 *
 * Used by: CliApp (replay subcommand).
 */

import fs from "node:fs";
import path from "node:path";
import type { ReplayOptions } from "../cli.types.js";
import type { Logger } from "../../shared/logger.js";
import type { CycleRunner } from "../../core/cycle.js";

/**
 * Default path to the trace.json file.
 */
const DEFAULT_TRACE_PATH = "trace.json";

/**
 * ReplayCommand class — re-executes the last agent run.
 *
 * Reads the last trace to extract input parameters and re-runs
 * the agent with the same configuration.
 *
 * Used by: CliApp (replay subcommand).
 */
export class ReplayCommand {
  /** Cycle runner for re-executing the agent. */
  private readonly cycleRunner: CycleRunner;

  /** Structured logger for output. */
  private readonly logger: Logger;

  /**
   * @param cycleRunner - Re-executes the agent with extracted parameters.
   * @param logger - Structured logger for output.
   */
  constructor(cycleRunner: CycleRunner, logger: Logger) {
    this.cycleRunner = cycleRunner;
    this.logger = logger;
  }

  /**
   * Replays the last execution with the same input parameters.
   *
   * @param options - CLI options for the replay command.
   *
   * Used by: CliApp (replay subcommand).
   *
   * Acceptance criteria:
   * - ReplayCommand re-executes with the same input.
   */
  async execute(options: ReplayOptions): Promise<void> {
    const tracePath = path.resolve(DEFAULT_TRACE_PATH);

    if (!fs.existsSync(tracePath)) {
      this.logger.error("No trace found. Run an agent first.");
      return;
    }

    const rawData = fs.readFileSync(tracePath, "utf-8");
    const dados = JSON.parse(rawData) as Record<string, unknown>;

    const entrada = dados['entrada'] as string | undefined;
    const tipo = dados['tipo_agente'] as string | undefined;
    const evento = dados['evento'] as string | undefined;

    if (!entrada) {
      this.logger.error("Trace does not contain input. Cannot replay.");
      return;
    }

    this.logger.info(`[replay] Re-executing with input: ${entrada}`);
    if (tipo) {
      this.logger.info(`[replay] Type: ${tipo}`);
    }
    if (evento) {
      this.logger.info(`[replay] Event: ${evento}`);
    }

    await this.cycleRunner.run({
      agentPath: options.agente,
      input: entrada,
      mode: tipo,
      event: evento,
    });
  }
}
