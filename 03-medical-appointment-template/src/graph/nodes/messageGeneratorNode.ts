import {
  getSystemPrompt,
  getUserPromptTemplate,
  MessageSchema,
} from "../../prompts/v1/messageGenerator.ts";
import { OpenRouterService } from "../../services/openRouterService.ts";
import type { GraphState } from "../graph.ts";
import { AIMessage } from "langchain";

export function createMessageGeneratorNode(llmClient: OpenRouterService) {
  return async ({
    actionSuccess,
    intent,
    professionalName,
    datetime,
    patientName,
    error,
    messages,
    actionError,
  }: GraphState): Promise<Partial<GraphState>> => {
    try {
      if (actionError) {
        return {
          messages: [...messages, new AIMessage(actionError)],
        };
      }

      const hasSucceded = actionSuccess ? "success" : "error";
      const scenario = `${intent ?? "unknown"}_${hasSucceded}`;
      const details = {
        professionalName,
        datetime,
        patientName,
        error,
      };

      const systemPrompt = getSystemPrompt();
      const userPrompt = getUserPromptTemplate({ scenario, details });
      const result = await llmClient.generateStructured(
        systemPrompt,
        userPrompt,
        MessageSchema,
      );

      if (result.error) {
        return {
          messages: [...messages, new AIMessage("Desculpe, errei.")],
        };
      }

      return {
        messages: [...messages, new AIMessage(result.data!.response)],
      };
    } catch (error) {
      return {
        messages: [
          ...messages,
          new AIMessage("An error occurred while processing your request."),
        ],
      };
    }
  };
}
