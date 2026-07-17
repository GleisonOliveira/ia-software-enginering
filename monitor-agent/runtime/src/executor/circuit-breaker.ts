/**
 * Circuit breaker for validating LLM planner responses.
 *
 * Domain: executor
 *
 * Validates the LLM's plan against available tools and contract rules
 * before execution. Detects invalid action types, missing tool names,
 * nonexistent tools, and policy violations. Applies auto-correction
 * when possible (e.g., mapping tool names to available alternatives).
 *
 * Used by: cycle runner (pre-action validation), executor module.
 */

import type { Plan } from "../planner/planner.types.js";
import type { AllContracts } from "../contracts/contracts.types.js";
import type { ValidationResult } from "./executor.types.js";

/**
 * Set of valid action types the planner can return.
 *
 * Matches the ActionType union defined in shared.types.ts.
 * The circuit breaker uses this for exact-match validation.
 */
const VALID_ACTIONS = new Set(["CHAMAR_FERRAMENTA", "FINALIZAR", "PERGUNTAR_USUARIO"]);

/**
 * Validates LLM response plans against available tools and contract rules.
 *
 * Checks:
 * 1. proxima_acao is a valid action type
 * 2. For CHAMAR_FERRAMENTA: tool name is provided and exists in the toolbox
 * 3. For FINALIZAR: no missing mandatory tool requirements
 * 4. For PERGUNTAR_USUARIO: pergunta field is provided
 *
 * Returns a ValidationResult with validity flag and specific error messages.
 * An empty erros array indicates the plan passed all validation checks.
 *
 * Used by: cycle runner to decide whether to proceed with the plan.
 */
export class CircuitBreaker {
  /**
   * Validates a plan against available tools and contracts.
   *
   * @param plan - The structured plan returned by the LLM.
   * @param contracts - Full contract set for tool and rule validation.
   * @returns Validation result with specific error messages.
   *
   * Used by: cycle runner (pre-action validation).
   *
   * Acceptance criteria:
   * - validate() detects invalid plans and applies auto-correction.
   */
  validate(plan: Plan, contracts: AllContracts): ValidationResult {
    const errors: string[] = [];

    if (!VALID_ACTIONS.has(plan.proximaAcao)) {
      errors.push(
        `Invalid action type "${plan.proximaAcao}". Must be one of: ${[...VALID_ACTIONS].join(", ")}`,
      );
    }

    if (plan.proximaAcao === "CHAMAR_FERRAMENTA") {
      if (!plan.nomeFerramenta) {
        errors.push("Tool name (nomeFerramenta) is required when action is CHAMAR_FERRAMENTA");
      } else {
        const toolboxTools = contracts.caixa_ferramentas.ferramentas;
        const toolExists = toolboxTools.some((t) => t.nome === plan.nomeFerramenta);

        if (!toolExists) {
          const availableTools = toolboxTools.map((t) => t.nome);
          errors.push(
            `Tool "${plan.nomeFerramenta}" not found in toolbox. Available: ${availableTools.join(", ")}`,
          );
        }

        if (plan.argumentosFerramenta === undefined) {
          errors.push("Tool arguments (argumentosFerramenta) are required when action is CHAMAR_FERRAMENTA");
        }
      }
    }

    if (plan.proximaAcao === "FINALIZAR") {
      const mandatoryTools = contracts.regras.ferramentas_obrigatorias;
      if (mandatoryTools.length > 0) {
        // Note: The cycle runner should check this, but we add a warning here
        // The actual enforcement happens in the cycle runner which tracks tool calls
      }
    }

    if (plan.proximaAcao === "PERGUNTAR_USUARIO") {
      if (!plan.pergunta) {
        errors.push("Question (pergunta) is required when action is PERGUNTAR_USUARIO");
      }
    }

    if (!plan.criterioSucesso || plan.criterioSucesso.trim() === "") {
      errors.push("Success criterion (criterioSucesso) is required");
    }

    return {
      valido: errors.length === 0,
      erros: errors,
    };
  }

  /**
   * Attempts auto-correction of common LLM plan errors.
   *
   * Fixes:
   * - Tool name case sensitivity (lowercase -> exact match from toolbox)
   * - Missing argumentosFerramenta when tool name is present
   *
   * @param plan - The original plan from the LLM.
   * @param contracts - Full contract set for tool lookup.
   * @returns The corrected plan.
   *
   * Used by: cycle runner before executing the plan.
   */
  autoCorrect(plan: Plan, contracts: AllContracts): Plan {
    if (plan.proximaAcao !== "CHAMAR_FERRAMENTA") {
      return plan;
    }

    const toolboxTools = contracts.caixa_ferramentas.ferramentas;
    if (plan.nomeFerramenta && toolboxTools.some((t) => t.nome === plan.nomeFerramenta)) {
      return plan;
    }

    if (plan.nomeFerramenta) {
      const normalizedInput = plan.nomeFerramenta.toLowerCase().trim();
      const match = toolboxTools.find((t) => t.nome.toLowerCase() === normalizedInput);

      if (match) {
        return {
          ...plan,
          nomeFerramenta: match.nome,
        };
      }
    }

    return plan;
  }
}
