/**
 * Lifecycle hook execution for the agent execution cycle.
 *
 * Domain: tools
 *
 * Executes hooks declared in the ganchos.md contract at specific points
 * in the execution cycle (before/after step, before/after action, on error).
 * Each hook action type (log, alerta) is handled differently:
 * - log: Writes a structured log message via the Logger.
 * - alerta: Emits a high-priority alert to stderr.
 *
 * Used by: cycle runner (lifecycle events), executor module.
 */

import type { AllContracts } from "../contracts/contracts.types.js";
import type { HookAction } from "../contracts/contracts.types.js";
import type { Logger } from "../shared/logger.js";

/**
 * Valid hook names matching the ganchos.md contract structure.
 *
 * Each name corresponds to a specific lifecycle event in the execution cycle.
 */
export type HookName =
  | "antes_da_etapa"
  | "apos_etapa"
  | "antes_da_acao"
  | "apos_acao"
  | "em_erro";

/**
 * Parameters passed to hook execution for context.
 *
 * Provides the step number and any relevant data for the hook to process.
 */
export interface HookParams {
  /** Current step number (etapa). */
  readonly etapa: number;
  /** Optional tool name (for action hooks). */
  readonly toolName?: string;
  /** Optional message or data to include in the hook output. */
  readonly message?: string;
}

/**
 * Executes lifecycle hooks defined in the contracts.
 *
 * Looks up hook actions from the ganchos.md contract by name,
 * then executes the corresponding action (log or alerta). If the
 * hook name is not found in the contract, the hook is silently skipped.
 *
 * Used by: cycle runner lifecycle events.
 */
export class HookExecutor {
  /** Structured logger for hook output. */
  private readonly logger: Logger;

  /**
   * @param logger - Structured logger for hook output.
   */
  constructor(logger: Logger) {
    this.logger = logger;
  }

  /**
   * Executes a lifecycle hook defined in the contracts.
   *
   * @param name - The hook name (e.g., "antes_da_etapa").
   * @param contracts - Full contract set containing hook definitions.
   * @param params - Context parameters for the hook execution.
   *
   * Used by: cycle runner lifecycle events.
   *
   * Acceptance criteria:
   * - Fires configured hooks.
   */
  execute(name: HookName, contracts: AllContracts, params: HookParams): void {
    const ganchos = contracts.ganchos.ganchos;
    const action: HookAction | undefined = ganchos[name];

    // Hook not configured — silently skip
    if (!action) {
      return;
    }

    const timestamp = new Date().toISOString().substring(11, 19);
    const detail = HookExecutor.formatDetail(name, params);

    switch (action) {
      case "log":
        this.logger.info(`[hook:${timestamp}] ${detail}`);
        break;
      case "alerta":
        // Alerts go to stderr for visibility in terminal output
        process.stderr.write(`[${timestamp}] [ALERT] ${detail}\n`);
        break;
      default: {
        // Exhaustive check — TypeScript will error if a case is missing
        const _exhaustive: never = action;
        this.logger.warn(`Unknown hook action: ${_exhaustive}`);
      }
    }
  }

  /**
   * Returns all hook names that have actions configured in the contracts.
   *
   * Useful for debugging and logging which hooks are active.
   *
   * @param contracts - Full contract set.
   * @returns Array of configured hook names.
   *
   * Used by: cycle runner initialization, debugging.
   */
  getConfigured(contracts: AllContracts): HookName[] {
    const ganchos = contracts.ganchos.ganchos;
    const allHooks: HookName[] = [
      "antes_da_etapa",
      "apos_etapa",
      "antes_da_acao",
      "apos_acao",
      "em_erro",
    ];

    return allHooks.filter((name) => ganchos[name] !== undefined);
  }

  /**
   * Formats hook detail string from the hook name and parameters.
   *
   * Creates a human-readable string describing the hook event and context.
   *
   * @param name - The hook name.
   * @param params - Hook execution parameters.
   * @returns Formatted detail string.
   *
   * Used by: execute().
   */
  private static formatDetail(name: HookName, params: HookParams): string {
    const parts: string[] = [`hook:${name}`, `step=${params.etapa}`];

    if (params.toolName) {
      parts.push(`tool=${params.toolName}`);
    }

    if (params.message) {
      parts.push(params.message);
    }

    return parts.join(" ");
  }
}
