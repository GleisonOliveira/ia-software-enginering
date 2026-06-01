import { END, MessagesValue, START, StateGraph, StateSchema } from "@langchain/langgraph";
import { z } from "zod/v4";
import { identifyIntent } from "./nodes/identifyIntentNode";
import { chatResponse } from "./nodes/chatResponseNode";
import { upperCaseNode } from "./nodes/upperCaseNode";
import { lowerCaseNode } from "./nodes/lowerCaseNode";
import { fallbackNode } from "./nodes/fallbackNode";

const stateSchema = new StateSchema({
  messages: MessagesValue,
  output: z.string(),
  command: z.enum(["uppercase", "lowercase", "unknown"]),
});

export type GraphState = typeof stateSchema;

export function buildGraph() {
  const graph = new StateGraph(stateSchema)
    .addNode("identifyIntent", identifyIntent)
    .addNode("chatResponse", chatResponse)
    .addNode("upperCaseNode", upperCaseNode)
    .addNode("lowerCaseNode", lowerCaseNode)
    .addNode("fallbackNode", fallbackNode)
    .addEdge(START, "identifyIntent")
    .addConditionalEdges(
      "identifyIntent",
      ({ command }) => {
        switch (command) {
          case "uppercase":
            return "uppercase";
          case "lowercase":
            return "lowercase";
          default:
            return "fallback";
        }
      },
      {
        uppercase: "upperCaseNode",
        lowercase: "lowerCaseNode",
        fallback: "fallbackNode",
      }
    )
    .addEdge("upperCaseNode", "chatResponse")
    .addEdge("lowerCaseNode", "chatResponse")
    .addEdge("fallbackNode", "chatResponse")
    .addEdge("chatResponse", END);

  return graph.compile();
}
