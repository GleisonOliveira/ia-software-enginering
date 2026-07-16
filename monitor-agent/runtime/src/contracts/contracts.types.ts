/**
 * TypeScript types derived from Zod contract schemas.
 *
 * Domain: contracts
 *
 * Uses `z.infer` to derive compile-time types from the Zod schemas defined
 * in schemas.ts. This ensures type safety is always in sync with runtime
 * validation — changing a schema automatically updates the corresponding type.
 */

import type { z } from "zod";
import type {
  AgentContractSchema,
  LoopContractSchema,
  PlannerContractSchema,
  ToolboxContractSchema,
  ExecutorContractSchema,
  RulesContractSchema,
  HooksContractSchema,
  SkillsContractSchema,
  MemoryContractSchema,
  AllContractsSchema,
  ParamTypeSchema,
  SkillParamSchema as SkillParamSchemaType,
  ToolboxToolSchema,
  SkillSchema,
  HookActionSchema,
} from "./schemas.js";

/**
 * Primitive type for tool parameters, inferred from ParamTypeSchema.
 *
 * Used by: SkillParam, payload validation, tool input/output definitions.
 */
export type ParamType = z.infer<typeof ParamTypeSchema>;

/**
 * Map of parameter names to their allowed types, inferred from SkillParamSchema.
 *
 * Used by: Skill, ToolboxTool, payload validation, tool builder.
 */
export type SkillParam = z.infer<typeof SkillParamSchemaType>;

/**
 * A single tool entry with name and input schema, inferred from ToolboxToolSchema.
 *
 * Used by: ToolboxContract, tool registry, payload validation.
 */
export type ToolboxTool = z.infer<typeof ToolboxToolSchema>;

/**
 * A skill definition with name, description, and input/output schemas.
 * Inferred from SkillSchema.
 *
 * Used by: SkillsContract, tool builder, tool registry.
 */
export type Skill = z.infer<typeof SkillSchema>;

/**
 * Hook action type (log or alerta), inferred from HookActionSchema.
 *
 * Used by: HooksContract, hook executor.
 */
export type HookAction = z.infer<typeof HookActionSchema>;

/**
 * Agent contract type with metadata, objective, and output contract.
 * Inferred from AgentContractSchema.
 *
 * Used by: AllContracts, contracts loader, cycle runner.
 */
export type AgentContract = z.infer<typeof AgentContractSchema>;

/**
 * Loop contract type with cycle limits and stop conditions.
 * Inferred from LoopContractSchema.
 *
 * Used by: AllContracts, cycle runner, state manager.
 */
export type LoopContract = z.infer<typeof LoopContractSchema>;

/**
 * Planner contract type with LLM output format and planning rules.
 * Inferred from PlannerContractSchema.
 *
 * Used by: AllContracts, planner, circuit breaker.
 */
export type PlannerContract = z.infer<typeof PlannerContractSchema>;

/**
 * Toolbox contract type listing available tools with input schemas.
 * Inferred from ToolboxContractSchema.
 *
 * Used by: AllContracts, tool registry, payload validation.
 */
export type ToolboxContract = z.infer<typeof ToolboxContractSchema>;

/**
 * Executor contract type with validation, retry, and evaluation settings.
 * Inferred from ExecutorContractSchema.
 *
 * Used by: AllContracts, executor module.
 */
export type ExecutorContract = z.infer<typeof ExecutorContractSchema>;

/**
 * Rules contract type with mandatory tools, limits, and policies.
 * Inferred from RulesContractSchema.
 *
 * Used by: AllContracts, cycle runner, state limits, circuit breaker.
 */
export type RulesContract = z.infer<typeof RulesContractSchema>;

/**
 * Hooks contract type with lifecycle hook actions.
 * Inferred from HooksContractSchema.
 *
 * Used by: AllContracts, hook executor.
 */
export type HooksContract = z.infer<typeof HooksContractSchema>;

/**
 * Skills contract type listing all available skills.
 * Inferred from SkillsContractSchema.
 *
 * Used by: AllContracts, tool builder, tool registry.
 */
export type SkillsContract = z.infer<typeof SkillsContractSchema>;

/**
 * Memory contract type with short-term and final summary configuration.
 * Inferred from MemoryContractSchema.
 *
 * Used by: AllContracts, memory manager.
 */
export type MemoryContract = z.infer<typeof MemoryContractSchema>;

/**
 * Composite type combining all 9 contract types.
 * Inferred from AllContractsSchema.
 *
 * Used by: contracts loader, state manager, cycle runner, all domain modules
 * that need access to contract data.
 */
export type AllContracts = z.infer<typeof AllContractsSchema>;
