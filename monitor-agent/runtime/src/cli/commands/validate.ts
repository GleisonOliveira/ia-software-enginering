/**
 * CLI command for validating agent contracts.
 *
 * Domain: cli
 *
 * Loads and validates all 9 contract files from an agent directory,
 * reporting success or listing validation errors. Useful for verifying
 * agent definitions before execution.
 *
 * Used by: CliApp (validate subcommand).
 */

import type { ValidateOptions } from "../cli.types.js";
import type { ContractLoader } from "../../contracts/loader.js";
import type { Logger } from "../../shared/logger.js";

/**
 * ValidateCommand class — validates agent contracts for completeness.
 *
 * Loads all contract files from the agent path and validates them
 * against Zod schemas. Reports success or lists specific errors.
 *
 * Used by: CliApp (validate subcommand).
 */
export class ValidateCommand {
  /** Contract loader for reading and validating agent files. */
  private readonly contractLoader: ContractLoader;

  /** Structured logger for output. */
  private readonly logger: Logger;

  /**
   * @param contractLoader - Loads and validates contracts from agent files.
   * @param logger - Structured logger for output.
   */
  constructor(contractLoader: ContractLoader, logger: Logger) {
    this.contractLoader = contractLoader;
    this.logger = logger;
  }

  /**
   * Validates contracts from the given agent path.
   *
   * Attempts to load all 9 contracts and validates them against Zod schemas.
   * Reports success or lists specific validation errors.
   *
   * @param options - CLI options for the validate command.
   *
   * Used by: CliApp (validate subcommand).
   *
   * Acceptance criteria:
   * - ValidateCommand validates contracts and returns ok/failure.
   */
  execute(options: ValidateOptions): void {
    this.logger.info(`Validating contracts from: ${options.agente}`);

    try {
      const contracts = this.contractLoader.loadAllContracts(options.agente);

      this.logger.info("Validation successful!");
      this.logger.info(`  Agent: ${contracts.agente.nome} (${contracts.agente.tipo})`);
      this.logger.info(`  Objective: ${contracts.agente.objetivo}`);
      this.logger.info(`  Tools: ${contracts.habilidades.habilidades.length}`);
      this.logger.info(`  Max steps: ${contracts.regras.limites.max_etapas}`);
      this.logger.info(`  Time limit: ${contracts.regras.limites.limite_tempo_segundos}s`);
      this.logger.info(`  Mandatory tools: ${contracts.regras.ferramentas_obrigatorias.join(", ") || "none"}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Validation failed: ${message}`);
      process.exit(1);
    }
  }
}
