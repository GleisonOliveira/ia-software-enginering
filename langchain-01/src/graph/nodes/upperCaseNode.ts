import { type GraphState } from "../graph";
import { GraphNode } from "@langchain/langgraph";

export const upperCaseNode: GraphNode<GraphState> = (state) => {
  const responseText = state.output.toUpperCase();

  return {
    ...state,
    output: responseText,
  };
};
