import { Neo4jService } from "../../services/neo4jService.ts";
import type { GraphState } from "../graph.ts";

async function executeQuery(query: string, neo4jService: Neo4jService) {
  try {
    const isValid = neo4jService.validateQuery(query);

    if (!isValid) {
      return {
        results: null,
        error: "Invalid query structure or invalid query",
      };
    }

    const results = await neo4jService.query(query);

    if (!results.length) {
      return {
        results: [],
        error: "No results found",
      };
    }

    return {
      results,
      error: null,
    };
  } catch (error) {
    return {
      results: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export function createCypherExecutorNode(neo4jService: Neo4jService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      const { results, error } = await executeQuery(state.query!, neo4jService);

      if (error && results === null) {
        return {
          ...state,
          error: "Invalid Cypher query - correction failed",
        };
      }

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
