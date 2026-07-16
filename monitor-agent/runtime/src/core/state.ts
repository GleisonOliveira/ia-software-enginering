/**
 * Agent state creation from contracts.
 *
 * Domain: core
 *
 * Creates the initial AgentState object from loaded contracts and user input.
 * Extracts limits from the regras.md contract, applies CLI overrides for agent
 * type, and initializes all mutable tracking fields to their zero values.
 *
 * Used by: Cycle runner entry point, replay command.
 */

import type { AllContracts } from "../contracts/contracts.types.js";
import type { AgentState, StateLimits } from "./state.types.js";
import type { AgentType, TokenUsage } from "../shared/shared.types.js";
import { EMPTY_TOKEN_USAGE } from "../shared/shared.types.js";

/**
 * Default maximum tokens the LLM may consume.
 *
 * Matches the Python runtime's hardcoded default (max_tokens = 50000).
 */
const DEFAULT_MAX_TOKENS = 50000;

/**
 * Default agent type when neither the contract nor CLI override specifies one.
 *
 * Matches the Python runtime's fallback (tipo_agente = "task_based").
 */
const DEFAULT_AGENT_TYPE: AgentType = "task_based";

/**
 * Extracts StateLimits from the regras.md contract with safe defaults.
 *
 * The regras.limites.chamadas_ferramenta field can be either a string "total"
 * (unlimited per-tool calls) or a map of tool names to integer limits.
 * Unrecognized values default to unlimited (Number.MAX_SAFE_INTEGER).
 *
 * @param contracts - The loaded and validated contract set.
 * @returns Resource limits for the execution cycle.
 *
 * Used by: createState().
 */
function extractLimits(contracts: AllContracts): StateLimits {
  const regras = contracts.regras;
  const limites = regras.limites;

  // Parse per-tool call limits from chamadas_ferramenta
  const limitesPorFerramenta: Record<string, number> = {};
  let maxChamadasFerramenta = 0;

  for (const [key, value] of Object.entries(limites.chamadas_ferramenta)) {
    if (typeof value === "number") {
      limitesPorFerramenta[key] = value;
      maxChamadasFerramenta += value;
    }
    // String values (e.g., "ilimitado") are treated as unlimited — no per-tool limit added
  }

  // If no per-tool limits were specified, set a generous total default
  if (maxChamadasFerramenta === 0) {
    maxChamadasFerramenta = limites.max_etapas * 3;
  }

  return {
    maxEtapas: limites.max_etapas,
    maxChamadasFerramenta,
    limitesPorFerramenta,
    semProgresso: limites.sem_progresso,
    limiteTempoSegundos: limites.limite_tempo_segundos,
    // Max tokens comes from the loop contract or falls back to env default
    maxTokens: DEFAULT_MAX_TOKENS,
  };
}

/**
 * Creates the initial AgentState from loaded contracts and user input.
 *
 * Initializes all mutable tracking fields (step counter, history, token usage)
 * to their zero values. The agent type is determined by the CLI override (mode)
 * falling back to the contract's tipo field, then to the default.
 *
 * @param contracts - The loaded and validated contract set.
 * @param input - User-provided input text for this execution run.
 * @param mode - Optional CLI override for the agent type.
 * @param event - Optional event context string.
 * @returns A fully initialized AgentState ready for the first cycle step.
 *
 * Used by: Cycle runner, replay command.
 *
 * Acceptance criteria:
 * - createState() returns an AgentState with all required fields populated.
 */
export function createState(
  contracts: AllContracts,
  input: string,
  mode?: string,
  event?: string,
): AgentState {
  // Determine agent type: CLI override > contract > default
  const tipoAgente = (mode as AgentType | undefined) ?? contracts.agente.tipo ?? DEFAULT_AGENT_TYPE;

  // Extract limits from regras.md with safe fallbacks
  const limits = extractLimits(contracts);

  // Initialize the zero-value token usage accumulator
  const tokensConsumidos: TokenUsage = { ...EMPTY_TOKEN_USAGE };

  return {
    // Contract-derived fields
    objetivo: contracts.agente.objetivo,
    entrada: input,
    tipoAgente,
    evento: event,

    // Step tracking (initialized to zero)
    etapa: 0,
    chamadasFerramenta: 0,
    chamadasPorFerramenta: {},

    // Resource limits
    limits,

    // Token tracking
    tokensConsumidos,

    // Safety: sensitive actions from regras.md
    acoesSensiveis: contracts.regras.acoes_sensiveis,

    // Execution state (initialized to empty/false)
    historico: [],
    concluido: false,
    resultado: "",
    etapasSemProgresso: 0,
    ultimaFerramenta: undefined,
  };
}
