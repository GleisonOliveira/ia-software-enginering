import { type GraphState } from "../graph";
import { GraphNode } from "@langchain/langgraph";

export const identifyIntent: GraphNode<GraphState> = (state) => {
  const input = state.messages.at(-1)?.text ?? "";

  let command: GraphState["State"]["command"] = "unknown";
  const inputLower = input.toLowerCase();

  if (inputLower.includes("upper")) {
    command = "uppercase";
  }

  if (inputLower.includes("lower")) {
    command = "lowercase";
  }

  return {
    ...state,
    command,
    output: input,
  };
};
