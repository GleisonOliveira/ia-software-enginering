import {
  getSystemPrompt,
  getUserPromptTemplate,
  QueryAnalysisSchema,
} from "../../prompts/v1/queryAnalyzer.ts";
import { OpenRouterService } from "../../services/openrouterService.ts";
import type { GraphState } from "../graph.ts";

export function createQueryPlannerNode(llmClient: OpenRouterService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      const systemPrompt = getSystemPrompt();
      const userPrompt = getUserPromptTemplate(state.question!);
      const result = await llmClient.generateStructured(
        systemPrompt,
        userPrompt,
        QueryAnalysisSchema,
      );

      if (!result.success) {
        return {
          error: result.error,
          isMultiStep: false,
        };
      }

      const {
        data: { requiresDecomposition, subQuestions },
      } = result;

      if (requiresDecomposition && subQuestions.length) {
        return {
          isMultiStep: true,
          subQuestions,
          currentStep: 0,
          subQueries: [],
          subResults: [],
        };
      }

      return {
        ...state,
      };
    } catch (error) {
      return {
        ...state,
        isMultiStep: false,
        error: `Failed to extract question: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  };
}
