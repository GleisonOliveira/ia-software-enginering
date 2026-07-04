import { AIMessage } from "langchain";
import { OpenRouterService } from "../../services/openRouterService.ts";
import type { GraphState } from "../state.ts";
import {
  getSystemPrompt,
  IntentData,
  IntentSchema,
} from "../../prompts/v1/identifyIntent.ts";

function isIntentData(
  result: string | IntentData | undefined,
): result is IntentData {
  return typeof result === "object" && !!result.fileType && !!result.intent;
}

export function intentNode(openRouterService: OpenRouterService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      const rawQuestion = state.messages.at(-1)!.text as string;
      const { data } = await openRouterService.generateStructured(
        getSystemPrompt(),
        rawQuestion,
        IntentSchema,
      );

      if (!isIntentData(data)) {
        throw new Error("Invalid intent data");
      }

      data.fileName ??= `data.${data.fileType}`;

      return {
        intent: data.intent,
        fileContent: data.fileContent ?? "",
        fileName: data.fileName,
      };
    } catch (error) {
      console.log(error);
      return {
        messages: [
          new AIMessage(
            "Sorry, I had trouble understanding the intent. Please rephrase your question or provide more details.",
          ),
        ],
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  };
}
