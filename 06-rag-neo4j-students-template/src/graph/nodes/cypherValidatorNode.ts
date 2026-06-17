import {
  CypherValidatorSchema,
  getSystemPrompt,
  getUserPromptTemplate,
} from "../../prompts/v1/cypherValidator.ts";
import { type OpenRouterService } from "../../services/openrouterService.ts";
import type { GraphState } from "../graph.ts";

export function createCypherValidatorNode(llmClient: OpenRouterService) {
  return async ({ query }: GraphState): Promise<Partial<GraphState>> => {
    try {
      if (!query) {
        return {
          secure: true,
        };
      }

      const systemPrompt = getSystemPrompt();
      const userPrompt = getUserPromptTemplate(query);

      const result = await llmClient.generateStructured(
        systemPrompt,
        userPrompt,
        CypherValidatorSchema,
      );

      if (!result.success) {
        return {
          secure: false,
          error: "Query can not be validated",
        };
      }

      const { analysis, secure } = result.data;

      if (!secure) {
        return {
          secure: false,
          error: analysis,
        };
      }

      return {
        secure: true,
      };
    } catch (error) {
      return {
        error: "Query can not be validated",
      };
    }
  };
}
