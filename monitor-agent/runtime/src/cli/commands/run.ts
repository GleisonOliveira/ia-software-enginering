/**
 * CLI command for running an agent.
 *
 * Domain: cli
 *
 * Orchestrates a single agent execution by loading contracts, building tools,
 * creating initial state, and running the full perceive→plan→act→evaluate cycle.
 * Accepts CLI options for agent path, input text, mode override, and event context.
 *
 * Used by: CliApp (run subcommand).
 */

import type { RunOptions } from "../cli.types.js";
import type { CycleRunner } from "../../core/cycle.js";
import type { Logger } from "../../shared/logger.js";

/**
 * RunCommand class — executes an agent with the given options.
 *
 * Loads contracts from the agent path, creates initial state, and delegates
 * to the CycleRunner for the actual execution loop.
 *
 * Used by: CliApp (run subcommand).
 */
export class RunCommand {
  /** Cycle runner for executing the perceive→plan→act→evaluate loop. */
  private readonly cycleRunner: CycleRunner;

  /** Structured logger for output. */
  private readonly logger: Logger;

  /**
   * @param cycleRunner - Executes the main execution cycle.
   * @param logger - Structured logger for output.
   */
  constructor(
    cycleRunner: CycleRunner,
    logger: Logger,
  ) {
    this.cycleRunner = cycleRunner;
    this.logger = logger;
  }

  /**
   * Executes the agent with the given CLI options.
   *
   * @param options - CLI options for the run command.
   * @returns The final agent state after execution.
   *
   * Used by: CliApp (run subcommand).
   *
   * Acceptance criteria:
   * - RunCommand executes an agent with --agente and --entrada.
   */
  async execute(options: RunOptions): Promise<void> {
    this.logger.info(`Executing agent from: ${options.agente}`);
    this.logger.info(`Input: ${options.entrada}`);

    if (options.modo) {
      this.logger.info(`Mode override: ${options.modo}`);
    }
    if (options.evento) {
      this.logger.info(`Event: ${options.evento}`);
    }

    await this.cycleRunner.run({
      agentPath: options.agente,
      input: options.entrada,
      mode: options.modo,
      event: options.evento,
    });
  }
}
