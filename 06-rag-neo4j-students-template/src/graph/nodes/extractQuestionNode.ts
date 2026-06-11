import type { GraphState } from "../graph.ts";

export function createExtractQuestionNode() {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      if (!state.messages?.length) {
        return {
          ...state,
          error: "No messages provided",
        };
      }

      const question = state.messages.at(-1)?.text ?? "";

      if (!question.trim()) {
        return {
          ...state,
          error: "No valid question found in messages",
        };
      }

      return {
        ...state,
        question,
      };
    } catch (error: any) {
      return {
        ...state,
        error: `Failed to extract question: ${error.message}`,
      };
    }
  };
}
