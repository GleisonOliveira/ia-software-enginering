import {
  getSystemPrompt,
  getUserPromptTemplate,
  IntentSchema,
} from "../../prompts/v1/identifyIntent.js";
import { professionals } from "../../services/appointmentService.js";
import { OpenRouterService } from "../../services/openRouterService.js";
import type { GraphState } from "../graph.ts";

export function createIdentifyIntentNode(llmClient: OpenRouterService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    const input = state.messages.at(-1)!.text;

    try {
      const systemPrompt = getSystemPrompt(professionals);
      const userPrompt = getUserPromptTemplate(input);
      const result = await llmClient.generateStructured(
        systemPrompt,
        userPrompt,
        IntentSchema,
      );

      if (!result.success) {
        return {
          intent: "unknown",
          error: result.error,
        };
      }

      return {
        ...result.data,
      };
    } catch (error) {
      return {
        ...state,
        intent: "unknown",
        error:
          error instanceof Error
            ? error.message
            : "Intent identification failed",
      };
    }
  };
}
