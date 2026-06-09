import type { Runtime } from "@langchain/langgraph";
import { type OpenRouterService } from "../../services/openrouterService.ts";
import type { GraphState } from "../graph.ts";
import {
  ChatResponseSchema,
  getSystemPrompt,
  getUserPromptTemplate,
} from "../../prompts/v1/chatResponse.ts";
import { AIMessage, HumanMessage } from "langchain";
import { type PreferencesService } from "../../services/preferencesService.ts";

export function createChatNode(
  llmClient: OpenRouterService,
  preferencesService: PreferencesService,
) {
  return async (
    state: GraphState,
    runtime?: Runtime,
  ): Promise<Partial<GraphState>> => {
    const userId = String(
      runtime?.context?.userId || state.userId || "unknowm",
    );

    const userContext =
      state.userContext ?? (await preferencesService.getBasicInfo(userId));
    const systemPrompt = getSystemPrompt(userContext);
    const conversationHistory = state.messages
      .map(
        (msg) =>
          `${HumanMessage.isInstance(msg) ? "User" : "AI"}: ${msg.content}`,
      )
      .join("\n");

    const userMessage = state.messages.at(-1)?.text ?? "";
    const userPrompt = getUserPromptTemplate(userMessage, conversationHistory);
    const result = await llmClient.generateStructured(
      systemPrompt,
      userPrompt,
      ChatResponseSchema,
    );

    if (!result.success || !result.data) {
      return {
        messages: [
          new AIMessage(
            "Desculpe encontrei um erro, por favor tente novamente",
          ),
        ],
      };
    }

    const { message, shouldSavePreferences, preferences } = result.data;
    const needsSummarization = state.messages.length >= 6;

    return {
      messages: [new AIMessage(message)],
      extractedPreferences: shouldSavePreferences ? preferences : undefined,
      needsSummarization,
    };
  };
}
