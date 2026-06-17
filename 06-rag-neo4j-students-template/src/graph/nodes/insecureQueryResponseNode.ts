import { AIMessage } from "langchain";
import { OpenRouterService } from "../../services/openrouterService.ts";
import type { GraphState } from "../graph.ts";
import {
  getSystemPrompt,
  getUserPromptTemplate,
  InsecureQueryResponseSchema,
} from "../../prompts/v1/insecureQueryResponse.ts";

export function createInsecureQueryResponseNode(llmClient: OpenRouterService) {
  return async ({ error }: GraphState): Promise<Partial<GraphState>> => {
    try {
      const systemPrompt = getSystemPrompt();
      const userPrompt = getUserPromptTemplate(
        error ?? "Não foi possível processar a busca",
      );

      const result = await llmClient.generateStructured(
        systemPrompt,
        userPrompt,
        InsecureQueryResponseSchema,
      );

      if (!result.success || !result.data.analysis) {
        return {
          messages: [new AIMessage("Não foi possível processar a busca")],
        };
      }

      return {
        messages: [new AIMessage(result.data.analysis)],
      };
    } catch (error: any) {
      return {
        messages: [new AIMessage("Não foi possível processar a busca")],
      };
    }
  };
}
