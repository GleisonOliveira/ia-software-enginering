/**
 * YAML extraction from .md contract files.
 *
 * Domain: contracts
 *
 * Loads agent contract definitions from .md files that contain fenced YAML
 * code blocks. Mirrors the Python runtime's carregar_yaml_do_md() function
 * using the same regex pattern for extraction, then validates against
 * Zod schemas for runtime safety.
 *
 * Used by: Cycle runner, state manager, planner, tool builder, all modules
 * that need contract data.
 */

import fs from "node:fs";
import path from "node:path";
import yaml from "js-yaml";
import { AllContractsSchema } from "./schemas.js";
import type { AllContracts } from "./contracts.types.js";
import type { Logger } from "../shared/logger.js";

/**
 * Regex pattern to extract the first fenced YAML block from a .md file.
 *
 * Matches ```yaml ... ``` with DOTALL flag (cross-line). The Python runtime
 * uses the same pattern (re.search with r"```yaml\n(.*?)```").
 * Only the first match is returned, matching Python behavior.
 */
const YAML_BLOCK_REGEX = /```yaml\n([\s\S]*?)```/;

/**
 * Maps contract keys to their expected .md filenames.
 *
 * Some contracts are at the agent root, others in a contracts/ subdirectory.
 * This mapping mirrors the Python runtime's file discovery logic in
 * carregar_contratos(). Filenames are kept in English where the Python
 * runtime used Portuguese names.
 */
const CONTRACT_FILE_MAP: Record<keyof AllContracts, string> = {
  agente: "agent.md",
  ciclo: "contracts/loop.md",
  planejador: "contracts/planner.md",
  caixa_ferramentas: "contracts/toolbox.md",
  executor: "contracts/executor.md",
  regras: "rules.md",
  ganchos: "hooks.md",
  habilidades: "skills.md",
  memoria: "memory.md",
};

/**
 * Loads agent contract definitions from .md files containing fenced YAML.
 *
 * Extracts YAML blocks from contract files, validates against Zod schemas,
 * and returns a fully typed AllContracts object. Handles missing files and
 * malformed YAML gracefully with logger warnings.
 *
 * Used by: Cycle runner, state manager, planner, tool builder.
 */
export class ContractLoader {
  /** Injected logger for file loading warnings and info messages. */
  private readonly logger: Logger;

  /**
   * @param logger - Logger instance for dependency injection (replaces module-level singleton).
   */
  constructor(logger: Logger) {
    this.logger = logger;
  }

  /**
   * Extracts the first YAML code block from a .md file and parses it.
   *
   * If the file does not exist or contains no YAML block, returns an empty
   * object. This matches the Python runtime's behavior of gracefully handling
   * missing or malformed contract files.
   *
   * @param filePath - Absolute or relative path to the .md file.
   * @returns Parsed YAML data as a record, or empty object if extraction fails.
   */
  loadYamlFromMd(filePath: string): Record<string, unknown> {
    const resolvedPath = path.resolve(filePath);

    if (!fs.existsSync(resolvedPath)) {
      this.logger.warn(`Contract file not found: ${resolvedPath}`);
      return {};
    }

    const content = fs.readFileSync(resolvedPath, "utf-8");
    const match = YAML_BLOCK_REGEX.exec(content);

    if (!match?.[1]) {
      this.logger.warn(`No YAML block found in: ${resolvedPath}`);
      return {};
    }

    try {
      const parsed = yaml.load(match[1]);
      // yaml.load returns null for empty documents
      return (parsed as Record<string, unknown>) ?? {};
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.logger.error(`Failed to parse YAML in ${resolvedPath}: ${message}`);
      return {};
    }
  }

  /**
   * Loads all 9 contract files from an agent directory and validates them
   * against the AllContracts Zod schema.
   *
   * Reads each .md file from the agent directory (or contracts/ subdirectory),
   * extracts the YAML, and validates the complete set. Throws if any required
   * contract is missing or invalid.
   *
   * @param agentPath - Path to the agent root directory containing contract .md files.
   * @returns Validated and typed AllContracts object.
   * @throws Error if contract loading or validation fails.
   *
   * Acceptance criteria:
   * - loadYamlFromMd() extracts YAML from a test .md file correctly
   * - loadAllContracts() loads all 9 contracts from a valid agent_path
   */
  loadAllContracts(agentPath: string): AllContracts {
    const resolvedAgentPath = path.resolve(agentPath);
    this.logger.info(`Loading contracts from: ${resolvedAgentPath}`);

    const rawContracts: Record<string, unknown> = {};

    for (const [key, filename] of Object.entries(CONTRACT_FILE_MAP)) {
      const filePath = path.join(resolvedAgentPath, filename);
      rawContracts[key] = this.loadYamlFromMd(filePath);
    }

    // Validate the complete set of contracts against the Zod schema
    const result = AllContractsSchema.safeParse(rawContracts);

    if (!result.success) {
      const issues = result.error.issues
        .map((i) => `  ${i.path.join(".")}: ${i.message}`)
        .join("\n");
      throw new Error(`Invalid contracts for agent at ${resolvedAgentPath}:\n${issues}`);
    }

    this.logger.info("All 9 contracts loaded and validated successfully");
    return result.data;
  }
}
