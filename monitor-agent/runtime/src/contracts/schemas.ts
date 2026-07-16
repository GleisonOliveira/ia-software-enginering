/**
 * Zod schemas for all YAML contract files loaded from agent definitions.
 *
 * Domain: contracts
 *
 * Each schema mirrors the structure of a `.md` contract file (agente, ciclo,
 * planejador, caixa_ferramentas, executor, regras, ganchos, habilidades, memoria).
 * Schemas serve dual purposes: runtime validation of parsed YAML and automatic
 * TypeScript type derivation via `z.infer` in contracts.types.ts.
 */

import { z } from "zod";

/**
 * Zod enum for tool parameter types. Maps to ParamType in shared.types.ts.
 *
 * Used by: SkillParamSchema, payload validation, tool input/output schemas.
 */
export const ParamTypeSchema = z.enum(["string", "int", "float", "bool", "list", "object"]);

/**
 * Zod schema for a skill parameter map: parameter names -> allowed types.
 *
 * Used by: SkillSchema, ToolboxToolSchema input/output definitions.
 */
export const SkillParamSchema = z.record(z.string(), ParamTypeSchema);

/**
 * Zod schema for the agente.md contract file.
 *
 * Validates agent metadata (name, type, objective) and output contract definition.
 * Used by: AllContractsSchema, contracts loader.
 */
export const AgentContractSchema = z.object({
  /** Agent display name. */
  nome: z.string(),
  /** Human-readable description of the agent's purpose. */
  descricao: z.string(),
  /** Agent type controlling runtime cycle behavior. */
  tipo: z.enum(["task_based", "interactive", "goal_oriented", "autonomous"]),
  /** High-level objective the agent should achieve. */
  objetivo: z.string(),
  /** Defines the expected output format and required fields. */
  contrato_saida: z.object({
    /** Output format: json, texto, or relatorio. */
    formato: z.enum(["json", "texto", "relatorio"]),
    /** Field names that must be present in the final output. */
    campos_obrigatorios: z.array(z.string()),
    /** Example output for reference. */
    exemplo: z.record(z.unknown()),
  }),
});

/**
 * Zod schema for the ciclo.md (loop) contract file.
 *
 * Defines the execution loop limits and stop conditions.
 * Used by: AllContractsSchema, cycle runner.
 */
export const LoopContractSchema = z.object({
  /** The objective to achieve across all cycles. */
  objetivo: z.string(),
  /** Cycle configuration limits. */
  ciclo: z.object({
    /** Maximum number of steps (etapas) before forced termination. */
    max_etapas: z.number().int().positive(),
  }),
  /** List of textual stop conditions checked by the evaluator. */
  condicoes_parada: z.array(z.string()),
});

/**
 * Zod schema for the planejador.md (planner) contract file.
 *
 * Defines the expected LLM output format and planning rules.
 * Used by: AllContractsSchema, planner, circuit breaker.
 */
export const PlannerContractSchema = z.object({
  /** Expected structure of the LLM's plan response. */
  formato_saida: z.object({
    /** Action type: CHAMAR_FERRAMENTA, FINALIZAR, or PERGUNTAR_USUARIO. */
    proxima_acao: z.string(),
    /** Tool name when proxima_acao is CHAMAR_FERRAMENTA. */
    nome_ferramenta: z.string().optional(),
    /** Tool arguments when proxima_acao is CHAMAR_FERRAMENTA. */
    argumentos_ferramenta: z.record(z.unknown()).optional(),
    /** Success criterion for this step. */
    criterio_sucesso: z.string(),
    /** Question to ask the user when proxima_acao is PERGUNTAR_USUARIO. */
    pergunta: z.string().optional(),
  }),
  /** Planning rules the LLM must follow. */
  regras: z.array(z.string()),
});

/**
 * Zod schema for a single tool entry in the caixa_ferramentas.md contract.
 *
 * Used by: ToolboxContractSchema, ToolRegistry.
 */
export const ToolboxToolSchema = z.object({
  /** Tool name (must match the skill contract's nome). */
  nome: z.string(),
  /** Input parameter schema: names -> types. */
  entrada: SkillParamSchema,
});

/**
 * Zod schema for the caixa_ferramentas.md (toolbox) contract file.
 *
 * Lists all tools available to the agent with their input schemas.
 * Used by: AllContractsSchema, tool registry, payload validation.
 */
export const ToolboxContractSchema = z.object({
  /** Array of available tool definitions. */
  ferramentas: z.array(ToolboxToolSchema),
});

/**
 * Zod schema for the executor.md contract file.
 *
 * Controls execution behavior: validation, retries, and post-execution evaluation.
 * Used by: AllContractsSchema, executor module.
 */
export const ExecutorContractSchema = z.object({
  /** Execution-phase configuration. */
  execucao: z.object({
    /** Whether to validate tool input arguments before execution. */
    validar_entrada: z.boolean(),
    /** Whether to retry failed tool calls. */
    tentar_novamente_em_falha: z.boolean(),
  }),
  /** Post-execution configuration. */
  pos_execucao: z.object({
    /** Whether to evaluate tool results against the success criterion. */
    avaliar_resultado: z.boolean(),
  }),
});

/**
 * Zod schema for the regras.md (rules) contract file.
 *
 * Defines mandatory tools, resource limits, sensitive actions, and policies.
 * Used by: AllContractsSchema, cycle runner, circuit breaker, state limits.
 */
export const RulesContractSchema = z.object({
  /** Tool names that must be called at least once during execution. */
  ferramentas_obrigatorias: z.array(z.string()),
  /** Resource limits enforced by the cycle runner. */
  limites: z.object({
    /** Maximum number of steps. */
    max_etapas: z.number().int().positive(),
    /** Steps without progress before stagnation detection triggers. */
    sem_progresso: z.number().int().nonnegative(),
    /** Total execution time limit in seconds. */
    limite_tempo_segundos: z.number().int().positive(),
    /** Per-tool call limits: tool name -> max calls (or "ilimitado"). */
    chamadas_ferramenta: z.record(z.union([z.string(), z.number().int().positive()])),
  }),
  /** Action names requiring human confirmation before execution. */
  acoes_sensiveis: z.array(z.string()),
  /** High-level policy rules for the LLM to follow. */
  politicas: z.array(z.string()),
});

/**
 * Zod enum for hook action types (log or alerta).
 *
 * Used by: HooksContractSchema, hook executor.
 */
export const HookActionSchema = z.enum(["log", "alerta"]);

/**
 * Zod schema for the ganchos.md (hooks) contract file.
 *
 * Defines lifecycle hooks that fire at specific points in the execution cycle.
 * Used by: AllContractsSchema, hook executor.
 */
export const HooksContractSchema = z.object({
  /** Hook actions for each lifecycle event. */
  ganchos: z.object({
    /** Fires before each step begins. */
    antes_da_etapa: HookActionSchema,
    /** Fires after each step completes. */
    apos_etapa: HookActionSchema,
    /** Fires before a tool action is executed. */
    antes_da_acao: HookActionSchema,
    /** Fires after a tool action completes. */
    apos_acao: HookActionSchema,
    /** Fires when an error occurs during execution. */
    em_erro: HookActionSchema,
  }),
});

/**
 * Zod schema for a single skill definition in the habilidades.md contract.
 *
 * Used by: SkillsContractSchema, tool builder, tool registry.
 */
export const SkillSchema = z.object({
  /** Skill name (unique identifier). */
  nome: z.string(),
  /** Human-readable description of what the skill does. */
  descricao: z.string(),
  /** Input parameter schema: names -> types. */
  entrada: SkillParamSchema,
  /** Output parameter schema: names -> types. */
  saida: SkillParamSchema,
});

/**
 * Zod schema for the habilidades.md (skills) contract file.
 *
 * Lists all available skills with their input/output schemas.
 * Used by: AllContractsSchema, tool builder, payload validation.
 */
export const SkillsContractSchema = z.object({
  /** Array of skill definitions. */
  habilidades: z.array(SkillSchema),
});

/**
 * Zod schema for the memoria.md (memory) contract file.
 *
 * Defines short-term memory rules and final summary structure.
 * Used by: AllContractsSchema, memory manager.
 */
export const MemoryContractSchema = z.object({
  /** Short-term memory configuration. */
  memoria_curta: z.object({
    /** Fields to retain across steps. */
    guardar: z.array(z.string()),
    /** Fields to discard after each step. */
    descartar: z.array(z.string()),
    /** Maximum number of memory records to keep. */
    max_registros: z.number().int().positive(),
  }),
  /** Final summary configuration. */
  resumo_final: z.object({
    /** Maximum lines in the summary. */
    max_linhas: z.number().int().positive(),
    /** Fields to include in the summary. */
    campos: z.array(z.string()),
  }),
});

/**
 * Composite Zod schema combining all 9 contract schemas.
 *
 * Used by: contracts loader to validate the complete set of loaded contracts.
 * Ensures all required contracts are present and valid before runtime starts.
 */
export const AllContractsSchema = z.object({
  agente: AgentContractSchema,
  ciclo: LoopContractSchema,
  planejador: PlannerContractSchema,
  caixa_ferramentas: ToolboxContractSchema,
  executor: ExecutorContractSchema,
  regras: RulesContractSchema,
  ganchos: HooksContractSchema,
  habilidades: SkillsContractSchema,
  memoria: MemoryContractSchema,
});
