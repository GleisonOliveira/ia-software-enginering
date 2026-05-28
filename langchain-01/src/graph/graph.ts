import { END, MessagesValue, START, StateGraph, StateSchema } from "@langchain/langgraph";
import { z } from "zod/v4";
import { identifyIntent } from "./nodes/identifyIntentNode";
import { chatResponse } from "./nodes/chatResponseNode";

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
    .addEdge(START, "identifyIntent")
    .addEdge("identifyIntent", "chatResponse")
    .addEdge("chatResponse", END);

  return graph.compile();
}
