/**
 * Type definitions for the tools domain.
 *
 * Domain: tools
 *
 * Defines ToolFunction (async function signature for executable tools),
 * ToolDefinition (metadata for tool lookup and registration), and
 * ToolRegistryEntry (association between a skill contract and its runtime definition).
 */

import type { ActionResult } from "../shared/shared.types.js";
import type { Skill } from "../contracts/contracts.types.js";

/**
 * Async function signature for executable tools.
 *
 * Each tool receives parsed arguments and returns an ActionResult.
 * Tools can be LLM-backed (using generateText) or local implementations.
 *
 * Used by: ToolDefinition.fn, tool builder, tool executor.
 */
export type ToolFunction = (args: Record<string, unknown>) => Promise<ActionResult>;

/**
 * Metadata for a registered tool, combining identity, schema, and execution function.
 *
 * Stored in the ToolRegistry for lookup by name during the act phase.
 *
 * Used by: ToolRegistry, tool builder, executor module.
 */
export interface ToolDefinition {
  /** Unique tool name matching the skill contract's nome. */
  readonly name: string;
  /** Human-readable description from the skill contract. */
  readonly description: string;
  /** Input parameter schema: names -> type strings. */
  readonly inputSchema: Record<string, string>;
  /** Output parameter schema: names -> type strings. */
  readonly outputSchema: Record<string, string>;
  /** The executable tool function. */
  readonly fn: ToolFunction;
}

/**
 * Associates a skill contract with its runtime ToolDefinition.
 *
 * Created by the tool builder when loading skills from the contracts.
 * Stored in the ToolRegistry for efficient lookup.
 *
 * Used by: tool builder, tool registry, tool executor.
 */
export interface ToolRegistryEntry {
  /** The original skill contract definition. */
  readonly skill: Skill;
  /** The runtime tool definition with executable function. */
  readonly definition: ToolDefinition;
}
