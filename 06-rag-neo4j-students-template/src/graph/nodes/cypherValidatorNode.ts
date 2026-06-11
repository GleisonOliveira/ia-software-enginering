import { Neo4jService } from "../../services/neo4jService.ts";
import { type OpenRouterService } from "../../services/openrouterService.ts";
import type { GraphState } from "../graph.ts";

async function executeQuery(query: string, neo4jService: Neo4jService) {
  try {
    const isValid = neo4jService.validateQuery(query);
  } catch (error) {}
}

export function createCypherValidatorNode(llmClient: OpenRouterService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      return {
        ...state,
      };
    } catch (error) {
      return {
        ...state,
        error: "Invalid Cypher query - correction failed",
      };
    }
  };
}
