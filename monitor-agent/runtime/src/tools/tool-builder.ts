/**
 * Tool builder — constructs executable tool functions from skill contracts.
 *
 * Domain: tools
 *
 * Converts skill contract definitions into executable ToolFunction instances.
 * Each tool can be LLM-backed (using the AI SDK for data generation) or use
 * a mock fallback when no API key is available. Mirrors the Python runtime's
 * construir_ferramenta() function.
 *
 * Used by: cycle runner initialization, tool registry population.
 */

import type { Skill } from "../contracts/contracts.types.js";
import type { ToolFunction, ToolDefinition, ToolRegistryEntry } from "./tools.types.js";
import type { ActionResult, TokenUsage } from "../shared/shared.types.js";
import { EMPTY_TOKEN_USAGE } from "../shared/shared.types.js";
import type { LlmClient } from "../llm/llm-client.js";
import type { Logger } from "../shared/logger.js";

/**
 * Zero-value token usage for fallback/mock tool results.
 *
 * When a tool uses mock data (no API key), token usage is zero.
 */
const MOCK_TOKEN_USAGE: TokenUsage = EMPTY_TOKEN_USAGE;

/**
 * Generates a fallback value for a tool output field based on its type.
 *
 * Used when the LLM is unavailable (no API key) to provide minimal
 * mock data that satisfies the output schema. Mirrors the Python
 * runtime's _gerar_valor_fallback() function.
 *
 * @param tipoCampo - The parameter type string (string, int, float, bool, list, object).
 * @param nomeCampo - The parameter name (used for generating descriptive values).
 * @returns A mock value matching the expected type.
 *
 * Used by: ToolBuilder.build() for mock fallback generation.
 */
function generateFallbackValue(tipoCampo: string, nomeCampo: string): unknown {
  const tipo = tipoCampo.toLowerCase();

  switch (tipo) {
    case "int":
      return 42;
    case "float":
      return 3.14;
    case "bool":
      return true;
    case "list":
      return [{ item: `${nomeCampo}_1` }, { item: `${nomeCampo}_2` }];
    case "object":
      return { campo: nomeCampo, valor: "fallback" };
    case "string":
    default:
      return `${nomeCampo}_fallback`;
  }
}

/**
 * Builds executable tool functions from skill contract definitions.
 *
 * Each tool can be LLM-backed (using the AI SDK for data generation) or
 * use a mock fallback when no API key is available.
 *
 * Used by: cycle runner initialization, tool registry population.
 */
export class ToolBuilder {
  /** Structured logger for warning output on LLM failures. */
  private readonly logger: Logger;

  /**
   * @param logger - Structured logger for warning output on LLM failures.
   */
  constructor(logger: Logger) {
    this.logger = logger;
  }

  /**
   * Builds an executable tool function from a skill contract definition.
   *
   * Creates a ToolFunction that attempts LLM-backed data generation first.
   * If no API key is available or the LLM call fails, falls back to
   * generating mock data using generateFallbackValue().
   *
   * @param skill - The skill contract definition.
   * @param llmClient - Optional LlmClient for LLM-backed tool execution.
   * @returns The executable tool function.
   *
   * Used by: tool builder, cycle runner tool initialization.
   *
   * Acceptance criteria:
   * - Returns an executable function from a skill definition.
   * - LLM-based tool execution with mock fallback.
   */
  build(skill: Skill, llmClient?: LlmClient): ToolFunction {
    const { nome, descricao, saida } = skill;

    // Build the system prompt for the LLM-backed tool
    const outputFields = Object.entries(saida)
      .map(([campo, tipo]) => `  - ${campo}: ${tipo}`)
      .join("\n");

    const systemPrompt = `You are a tool called '${nome}'.
Function: ${descricao}

You MUST return ONLY valid JSON with exactly these fields:
${outputFields}

Rules:
- Generate realistic data consistent with the provided arguments
- For 'list' fields, return a list of objects with realistic details
- For 'object' fields, return a structured object with real data
- For 'string' fields, return descriptive and specific text
- NEVER use placeholders like 'mock', 'example', 'test' — generate real content
- Data must be consistent with each other and the provided context
- Respond in Portuguese`;

    const logger = this.logger;

    const toolFn: ToolFunction = async (args: Record<string, unknown>): Promise<ActionResult> => {
      // Attempt LLM-backed execution if a client is available
      if (llmClient) {
        try {
          const userPrompt = `Arguments received:\n${JSON.stringify(args, null, 2)}`;
          const response = await llmClient.callLlm({
            systemPrompt,
            userPrompt,
          });

          // Try to parse the LLM response as JSON
          const parsed = JSON.parse(response.text) as Record<string, unknown>;

          return {
            sucesso: true,
            dados: parsed,
            erro: "",
            _tokens: {
              prompt: response.usage.promptTokens,
              completion: response.usage.completionTokens,
              total: response.usage.totalTokens,
            },
            _entrada: args,
          };
        } catch (error) {
          logger.warn(`LLM-backed tool "${nome}" failed, using fallback`, {
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }

      // Fallback: generate mock data from the output schema
      const dados: Record<string, unknown> = {};
      for (const [nomeCampo, tipoCampo] of Object.entries(saida)) {
        dados[nomeCampo] = generateFallbackValue(tipoCampo, nomeCampo);
      }

      return {
        sucesso: true,
        dados,
        erro: "",
        _tokens: MOCK_TOKEN_USAGE,
        _entrada: args,
      };
    };

    return toolFn;
  }

  /**
   * Builds a ToolRegistryEntry from a skill contract definition.
   *
   * Creates the full entry including metadata (name, description, schemas)
   * and the executable function, suitable for registration in the ToolRegistry.
   *
   * @param skill - The skill contract definition.
   * @param llmClient - Optional LlmClient for LLM-backed tool execution.
   * @returns A complete ToolRegistryEntry.
   *
   * Used by: tool builder initialization, cycle runner.
   */
  buildEntry(skill: Skill, llmClient?: LlmClient): ToolRegistryEntry {
    const fn = this.build(skill, llmClient);

    const definition: ToolDefinition = {
      name: skill.nome,
      description: skill.descricao,
      inputSchema: skill.entrada,
      outputSchema: skill.saida,
      fn,
    };

    return {
      skill,
      definition,
    };
  }
}
