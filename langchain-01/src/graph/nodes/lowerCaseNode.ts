import { type GraphState } from "../graph";
import { GraphNode } from "@langchain/langgraph";

export const lowerCaseNode: GraphNode<GraphState> = (state) => {
  const responseText = state.output.toLowerCase();

  return {
    ...state,
    output: responseText,
  };
};
