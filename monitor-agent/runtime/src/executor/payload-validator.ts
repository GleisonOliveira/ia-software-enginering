/**
 * Payload validation for tool input arguments.
 *
 * Domain: executor
 *
 * Validates tool arguments against the skill contract's input schema
 * before execution. Checks for missing required fields and type mismatches.
 * Mirrors the Python runtime's validar_payload() function.
 *
 * Used by: executor module, cycle runner (pre-execution validation).
 */

import type { AllContracts } from "../contracts/contracts.types.js";
import type { SkillParam } from "../contracts/contracts.types.js";

/**
 * Maps contract parameter type strings to JavaScript type constructors.
 *
 * Used by validate() to perform runtime type checking on tool arguments.
 * The "float" type accepts both int and float to match the Python runtime's behavior
 * where YAML parses numbers as int or float depending on the value.
 */
const TYPE_MAP: Record<string, (value: unknown) => boolean> = {
  string: (v): boolean => typeof v === "string",
  int: (v): boolean => typeof v === "number" && Number.isInteger(v),
  float: (v): boolean => typeof v === "number",
  bool: (v): boolean => typeof v === "boolean",
  list: (v): boolean => Array.isArray(v),
  object: (v): boolean => v !== null && typeof v === "object" && !Array.isArray(v),
};

/**
 * Validates tool input/output arguments against skill contract schemas.
 *
 * Provides both input (pre-execution) and output (post-execution) validation.
 * Returns error lists rather than throwing, allowing callers to handle
 * validation failures gracefully.
 *
 * Used by: ToolExecutor, Evaluator, CycleRunner.
 */
export class PayloadValidator {
  /**
   * Validates tool input arguments against the skill contract's input schema.
   *
   * Checks that all required fields from the schema are present in the arguments
   * and that their values match the expected types.
   *
   * @param toolName - Name of the tool being validated.
   * @param args - Arguments provided by the LLM planner.
   * @param contracts - Full contract set containing skill definitions.
   * @returns Array of validation error messages (empty = valid payload).
   *
   * Used by: ToolExecutor (pre-execution validation).
   *
   * Acceptance criteria:
   * - Returns error list for invalid args (empty list for valid args).
   */
  validate(
    toolName: string,
    args: Record<string, unknown>,
    contracts: AllContracts,
  ): string[] {
    const errors: string[] = [];

    const habilidades = contracts.habilidades.habilidades;
    const skill = habilidades.find((h) => h.nome === toolName);

    if (!skill) {
      return [`Tool "${toolName}" not found in skills contract`];
    }

    const schemaEntrada: SkillParam = skill.entrada;
    const safeArgs = args ?? {};

    for (const [campo, tipoEsperado] of Object.entries(schemaEntrada)) {
      if (!(campo in safeArgs)) {
        errors.push(`Required field "${campo}" is missing`);
        continue;
      }

      const valor = safeArgs[campo];
      const tipoNormalizado = (typeof tipoEsperado === "string" ? tipoEsperado : "string").toLowerCase();
      const typeCheck = TYPE_MAP[tipoNormalizado];

      if (typeCheck && valor !== null && valor !== undefined && !typeCheck(valor)) {
        errors.push(
          `Field "${campo}": expected ${tipoNormalizado}, received ${typeof valor}`,
        );
      }
    }

    return errors;
  }

  /**
   * Validates tool output data against the skill contract's output schema.
   *
   * Checks that all required output fields are present and non-empty.
   * Only validates when the tool execution was successful (sucesso=true).
   *
   * @param toolName - Name of the tool whose output is being validated.
   * @param result - The ActionResult returned by the tool.
   * @param contracts - Full contract set containing skill definitions.
   * @returns Array of output validation problems (empty = valid output).
   *
   * Used by: Evaluator for post-execution output validation.
   */
  validateOutput(
    toolName: string,
    result: { sucesso: boolean; dados: Record<string, unknown> },
    contracts: AllContracts,
  ): string[] {
    const problems: string[] = [];

    if (!result.sucesso) {
      return problems;
    }

    const dados = result.dados;
    const habilidades = contracts.habilidades.habilidades;
    const skill = habilidades.find((h) => h.nome === toolName);

    if (!skill) {
      return problems;
    }

    const schemaSaida: SkillParam = skill.saida;

    for (const [campo, tipoEsperado] of Object.entries(schemaSaida)) {
      if (!(campo in dados)) {
        problems.push(`Output field "${campo}" is missing from result`);
        continue;
      }

      const valor = dados[campo];
      const tipoNormalizado = (typeof tipoEsperado === "string" ? tipoEsperado : "string").toLowerCase();

      if (valor === null || valor === undefined) {
        problems.push(`Output field "${campo}" is null/undefined`);
      } else if (typeof valor === "string" && valor.trim() === "") {
        problems.push(`Output field "${campo}" is an empty string`);
      } else if (Array.isArray(valor) && valor.length === 0) {
        problems.push(`Output field "${campo}" is an empty array`);
      }

      const typeCheck = TYPE_MAP[tipoNormalizado];
      if (typeCheck && valor !== null && valor !== undefined && !typeCheck(valor)) {
        problems.push(
          `Output field "${campo}": expected ${tipoNormalizado}, received ${typeof valor}`,
        );
      }
    }

    return problems;
  }
}
