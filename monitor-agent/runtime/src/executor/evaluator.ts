/**
 * Post-action evaluation of tool execution results.
 *
 * Domain: executor
 *
 * Evaluates whether each step achieved its success criterion by checking
 * the plan's action type, the tool's execution result, and output schema
 * compliance. Returns an Evaluation with quality rating and objective status.
 *
 * Used by: cycle runner (evaluate phase), evaluator module.
 */

import type { Plan } from "../planner/planner.types.js";
import type { Evaluation } from "../core/cycle.types.js";
import type { AllContracts } from "../contracts/contracts.types.js";
import type { ActionResult, QualityRating } from "../shared/shared.types.js";
import type { PayloadValidator } from "./payload-validator.js";

/**
 * Evaluates tool execution results against success criteria.
 *
 * The evaluation logic depends on the plan's action type:
 *
 * - FINALIZAR: Immediately returns objetivoAlcancado=true with the
 *   success criterion as the reason.
 *
 * - CHAMAR_FERRAMENTA: Checks if the tool execution was successful,
 *   validates output against the skill contract's output schema,
 *   and assigns a quality rating (completa/parcial/falha).
 *
 * - PERGUNTAR_USUARIO: Always considered complete (user was asked).
 *
 * Used by: cycle runner (evaluate phase).
 */
export class Evaluator {
  /** Payload validator for output schema validation. */
  private readonly payloadValidator: PayloadValidator;

  /**
   * @param payloadValidator - Validates tool output against skill schemas.
   */
  constructor(payloadValidator: PayloadValidator) {
    this.payloadValidator = payloadValidator;
  }

  /**
   * Evaluates the result of a tool execution step.
   *
   * @param plan - The LLM-generated plan for this step.
   * @param actionResult - The ActionResult from tool execution (undefined for non-tool actions).
   * @param contracts - Full contract set for output validation.
   * @returns Evaluation with objective status, quality rating, and any output problems.
   *
   * Used by: cycle runner (evaluate phase).
   *
   * Acceptance criteria:
   * - Returns correct objetivo_alcancado and qualidade.
   */
  evaluate(
    plan: Plan,
    actionResult: ActionResult | undefined,
    contracts: AllContracts,
  ): Evaluation {
    // FINALIZAR action: objective is achieved by declaration
    if (plan.proximaAcao === "FINALIZAR") {
      return {
        objetivoAlcancado: true,
        motivo: plan.criterioSucesso || "Objective achieved (FINALIZAR)",
        qualidade: undefined,
        problemasSaida: [],
      };
    }

    // PERGUNTAR_USUARIO action: question was asked, step is complete
    if (plan.proximaAcao === "PERGUNTAR_USUARIO") {
      return {
        objetivoAlcancado: false,
        motivo: `Question asked: ${plan.pergunta ?? "no question"}`,
        qualidade: undefined,
        problemasSaida: [],
      };
    }

    // CHAMAR_FERRAMENTA action: evaluate the tool execution result
    if (!actionResult) {
      return {
        objetivoAlcancado: false,
        motivo: "No action result — tool was not executed",
        qualidade: "falha",
        problemasSaida: [],
      };
    }

    // Tool execution failed
    if (!actionResult.sucesso) {
      return {
        objetivoAlcancado: false,
        motivo: `Step failed — ${actionResult.erro || "no error details"}`,
        qualidade: "falha",
        problemasSaida: [],
      };
    }

    // Tool execution succeeded — validate output against schema
    const nomeFerramenta = plan.nomeFerramenta ?? "";
    const problemasSaida = contracts.executor.pos_execucao.avaliar_resultado
      ? this.payloadValidator.validateOutput(nomeFerramenta, actionResult, contracts)
      : [];

    // Determine quality rating based on output validation
    let qualidade: QualityRating;
    let motivo: string;

    if (problemasSaida.length > 0) {
      qualidade = "parcial";
      motivo = `Step OK with caveats — ${problemasSaida.join("; ")}`;
    } else {
      qualidade = "completa";
      const criterio = plan.criterioSucesso;
      motivo = criterio ? `Step OK — criterion: ${criterio}` : "Step OK — continue";
    }

    return {
      objetivoAlcancado: false,
      motivo,
      qualidade,
      problemasSaida,
    };
  }
}
