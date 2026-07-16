/**
 * Type definitions for the CLI command options.
 *
 * Domain: cli
 *
 * Defines option interfaces for each Commander.js subcommand (run, validate,
 * trace, analyze, replay). These types ensure CLI argument parsing stays in
 * sync with the actual command implementations.
 */

import type { AgentType } from "../shared/shared.types.js";

/**
 * CLI options for the `run` command.
 *
 * Maps directly to Commander.js option definitions in commands/run.ts.
 *
 * Used by: run command handler, cycle runner entry point.
 */
export interface RunOptions {
  /** Path to the agent directory containing contract files. */
  readonly agente: string;
  /** User input text for the agent to process. */
  readonly entrada: string;
  /** Optional agent type override (defaults to contract's tipo). */
  readonly modo?: AgentType;
  /** Optional event context for the execution. */
  readonly evento?: string;
}

/**
 * CLI options for the `validate` command.
 *
 * Used by: validate command handler, contracts loader.
 */
export interface ValidateOptions {
  /** Path to the agent directory to validate. */
  readonly agente: string;
}

/**
 * CLI options for the `trace` command.
 *
 * Used by: trace command handler.
 */
export interface TraceOptions {
  /** Optional path to a specific trace file; defaults to the most recent. */
  readonly arquivo?: string;
}

/**
 * CLI options for the `analyze` command.
 *
 * Used by: analyze command handler.
 */
export interface AnalyzeOptions {
  /** Path to the agent directory. */
  readonly agente: string;
  /** Optional path to a specific trace file to analyze. */
  readonly trace?: string;
}

/**
 * CLI options for the `replay` command.
 *
 * Used by: replay command handler, cycle runner replay function.
 */
export interface ReplayOptions {
  /** Path to the agent directory to replay. */
  readonly agente: string;
}
