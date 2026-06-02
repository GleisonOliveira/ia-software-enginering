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
  }: GraphState): Promise<Partial<GraphState>> => {
    try {
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

      if (result.error || !result.data) {
        return {
          messages: [...messages, new AIMessage("Desculpe, errei.")],
        };
      }

      return {
        messages: [...messages, new AIMessage(result.data.message)],
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
