import { OpenRouterService } from "../../services/openrouterService.ts";
import { Neo4jService } from "../../services/neo4jService.ts";
import type { GraphState } from "../graph.ts";
import {
  CypherCorrectionSchema,
  getSystemPrompt,
  getUserPromptTemplate,
} from "../../prompts/v1/cypherCorrection.ts";

export function createCypherCorrectionNode(
  llmClient: OpenRouterService,
  neo4jService: Neo4jService,
) {
  return async ({
    query,
    validationError,
    question,
    originalQuery,
    correctionAttempts,
  }: GraphState): Promise<Partial<GraphState>> => {
    try {
      const schema = await neo4jService.getSchema();
      const systemPrompt = getSystemPrompt(schema);
      const userPrompt = getUserPromptTemplate(
        query!,
        validationError!,
        question,
      );

      const result = await llmClient.generateStructured(
        systemPrompt,
        userPrompt,
        CypherCorrectionSchema,
      );

      if (!result.success) {
        return {
          error: `Query correction failed: ${result.error ?? "Unknown error"}`,
        };
      }

      const { correctedQuery } = result.data;

      return {
        query: correctedQuery,
        originalQuery: originalQuery ?? query,
        correctionAttempts: (correctionAttempts ?? 0) + 1,
        validationError: undefined,
        needsCorrection: false,
      };
    } catch (error: any) {
      return {
        error: `Query correction failed: ${error.message}`,
      };
    }
  };
}
