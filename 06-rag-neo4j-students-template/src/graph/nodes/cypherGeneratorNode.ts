import { OpenRouterService } from "../../services/openrouterService.ts";
import { Neo4jService } from "../../services/neo4jService.ts";
import type { GraphState } from "../graph.ts";
import {
  CypherQuerySchema,
  getSystemPrompt,
  getUserPromptTemplate,
} from "../../prompts/v1/cypherGenerator.ts";
import { SALES_CONTEXT } from "../../prompts/v1/salesContext.ts";

function getCurrentStepQuestion(state: GraphState) {
  if (
    !state.isMultiStep ||
    !state.subQuestions?.length ||
    state.currentStep === undefined
  ) {
    return null;
  }

  if (state.currentStep >= state.subQuestions.length) {
    return null;
  }

  return {
    question: state.subQuestions[state.currentStep],
    stepNumber: state.currentStep + 1,
  };
}

export function createCypherGeneratorNode(
  llmClient: OpenRouterService,
  neo4jService: Neo4jService,
) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      const stepInfo = getCurrentStepQuestion(state);
      const targetQuestion = stepInfo?.question ?? state.question!;
      const schema = await neo4jService.getSchema();
      const systemPrompt = await getSystemPrompt(schema, SALES_CONTEXT);
      const userPrompt = await getUserPromptTemplate(targetQuestion);

      const result = await llmClient.generateStructured(
        systemPrompt,
        userPrompt,
        CypherQuerySchema,
      );

      if (!result.success) {
        return {
          error: `Failed to generate query: ${result.error ?? "Unknown error"}`,
        };
      }

      const {
        data: { query },
      } = result;

      if (state.isMultiStep && state.subQueries?.length) {
        return {
          query,
          subQueries: [...state.subQueries, query],
        };
      }

      return {
        query,
      };
    } catch (error: any) {
      return {
        ...state,
        error: `Failed to generate query: ${error.message}`,
      };
    }
  };
}
