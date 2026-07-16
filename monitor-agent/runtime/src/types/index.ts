/**
 * Public barrel export for all runtime types and schemas.
 *
 * Domain: core (barrel)
 *
 * Re-exports every type and value consumers need from a single entry point.
 * Import from `types/index.ts` instead of reaching into individual domain
 * modules to keep import paths stable across refactors.
 */

export type {
  AgentType,
  ParamType as SharedParamType,
  ActionType,
  QualityRating,
  HookAction as SharedHookAction,
  OutputFormat,
  TokenUsage,
  ActionResult,
  SkillParamSchema,
} from "../shared/shared.types.js";

export { EMPTY_TOKEN_USAGE } from "../shared/shared.types.js";

export type {
  LlmProvider,
  LlmConfig,
  CallLlmOptions,
  LlmResponse,
  LlmUsage,
  StructuredOutputOptions,
} from "../llm/llm.types.js";

export type {
  AgentContract,
  LoopContract,
  PlannerContract,
  ToolboxContract,
  ExecutorContract,
  RulesContract,
  HooksContract,
  SkillsContract,
  MemoryContract,
  AllContracts,
  ParamType,
  SkillParam,
  ToolboxTool,
  Skill,
  HookAction,
} from "../contracts/contracts.types.js";

export {
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
  SkillParamSchema as SkillParamZodSchema,
  ToolboxToolSchema,
  SkillSchema,
  HookActionSchema,
} from "../contracts/schemas.js";

export type {
  AgentState,
  StateLimits,
  MutableAgentState,
} from "../core/state.types.js";

export type {
  CycleConfig,
  Evaluation,
  HistoryEntry,
  Trace,
  ValidActionType,
} from "../core/cycle.types.js";

export type {
  ValidationResult,
  ExecutionResult,
  ToolExecutorContext,
  ToolInputSchema,
} from "../executor/executor.types.js";

export type {
  ToolFunction,
  ToolDefinition,
  ToolRegistryEntry,
} from "../tools/tools.types.js";

export type {
  Plan,
  Perception,
  PlannerContext,
} from "../planner/planner.types.js";

export type {
  TelemetryEventType,
  PhaseName,
  TelemetryEvent,
  PhaseMarker,
  PhaseStats,
  HealthMetrics,
  PerformanceData,
  TelemetrySummary,
} from "../telemetry/telemetry.types.js";

export type {
  RunOptions,
  ValidateOptions,
  TraceOptions,
  AnalyzeOptions,
  ReplayOptions,
} from "../cli/cli.types.js";
