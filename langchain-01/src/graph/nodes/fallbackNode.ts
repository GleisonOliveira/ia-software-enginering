import { AIMessage } from "langchain";
import { type GraphState } from "../graph";
import { GraphNode } from "@langchain/langgraph";

export const fallbackNode: GraphNode<GraphState> = (state) => {
  const responseText =
    "Unknown command, try 'uppercase', 'lowercase' or 'convert to lowercase'or 'make uppercase'";
  const fallbackMessage = new AIMessage(responseText).content.toString();

  return {
    ...state,
    output: fallbackMessage,
  };
};
