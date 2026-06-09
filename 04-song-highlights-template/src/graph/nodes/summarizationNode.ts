import { HumanMessage } from "langchain";
import { OpenRouterService } from "../../services/openrouterService.ts";
import type { GraphState } from "../graph.ts";
import {
  getSummarizationSystemPrompt,
  getSummarizationUserPrompt,
  SummarySchema,
} from "../../prompts/v1/summarization.ts";
import { type Runtime } from "@langchain/langgraph";
import { type PreferencesService } from "../../services/preferencesService.ts";
import { RemoveMessage } from "@langchain/core/messages";

export function createSummarizationNode(
  llmClient: OpenRouterService,
  preferencesService: PreferencesService,
) {
  return async (
    state: GraphState,
    runtime: Runtime,
  ): Promise<Partial<GraphState>> => {
    const conversationHistory = state.messages.map((msg) => ({
      role: HumanMessage.isInstance(msg) ? "User" : "AI",
      content: msg.text,
    }));

    const previousSummary = state.conversationSummary ?? undefined;
    const systemPrompt = getSummarizationSystemPrompt();
    const userPrompt = getSummarizationUserPrompt(
      conversationHistory,
      previousSummary,
    );

    const result = await llmClient.generateStructured(
      systemPrompt,
      userPrompt,
      SummarySchema,
    );

    if (result.error || !result.data) {
      return {
        needsSummarization: false,
      };
    }

    const userId = String(
      runtime?.context?.userId || state.userId || "unknowm",
    );

    await preferencesService.storeSummary(userId, result.data);

    const deleteMessage = state.messages
      .slice(0, 2)
      .map((msg) => new RemoveMessage({ id: msg.id as string }));

    return {
      messages: deleteMessage,
      conversationSummary: result.data,
      needsSummarization: false,
    };
  };
}
