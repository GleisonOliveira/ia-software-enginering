import { AIMessage } from "langchain";
import { type GraphState } from "../graph";
import { GraphNode } from "@langchain/langgraph";

export const chatResponse: GraphNode<GraphState> = (state) => {
  const responseText = state.output;
  const aiMessage = new AIMessage(responseText);

  return {
    ...state,
    messages: [...state.messages, aiMessage],
  };
};
