import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import type { OpenRouterService } from "../../../src/services/openrouterService.ts";
import type { Neo4jService } from "../../../src/services/neo4jService.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import { createCypherCorrectionNode } from "../../../src/graph/nodes/cypherCorrectionNode.ts";

function buildState(overrides?: Partial<GraphState>): GraphState {
  return {
    messages: [],
    query: "MATCH (c:Course) RETURN c.name",
    validationError:
      "Syntax error: Invalid Cypher query near 'MATCH (c:Course)'",
    ...overrides,
  } as GraphState;
}

describe("createCypherCorrectionNode", () => {
  let mockLLMClient: jest.Mocked<OpenRouterService>;
  let mockNeo4jService: jest.Mocked<Neo4jService>;
  let cypherCorrectionNode: ReturnType<typeof createCypherCorrectionNode>;

  const mockSchema = "Node properties: Course {name: string, url: string}";
  const correctedQuery = "MATCH (c:Course) RETURN c.name AS courseName";

  beforeEach(() => {
    mockLLMClient = {
      generateStructured: jest.fn(),
    } as unknown as jest.Mocked<OpenRouterService>;

    mockNeo4jService = {
      getSchema: jest
        .fn<() => Promise<string>>()
        .mockResolvedValue(mockSchema),
    } as unknown as jest.Mocked<Neo4jService>;

    cypherCorrectionNode = createCypherCorrectionNode(
      mockLLMClient,
      mockNeo4jService,
    );
  });

  it("should return corrected query on success", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { correctedQuery, explanation: "Fixed syntax error" },
    });

    const result = await cypherCorrectionNode(buildState());

    expect(result.query).toBe(correctedQuery);
    expect(result.needsCorrection).toBe(false);
    expect(result.validationError).toBeUndefined();
  });

  it("should increment correctionAttempts", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { correctedQuery, explanation: "Fixed syntax error" },
    });

    const result = await cypherCorrectionNode(
      buildState({ correctionAttempts: 2 }),
    );

    expect(result.correctionAttempts).toBe(3);
  });

  it("should use originalQuery from state when available", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { correctedQuery, explanation: "Fixed syntax error" },
    });

    const result = await cypherCorrectionNode(
      buildState({ originalQuery: "MATCH (c:Course) RETURN n" }),
    );

    expect(result.originalQuery).toBe("MATCH (c:Course) RETURN n");
  });

  it("should fallback to query as originalQuery when originalQuery is not set", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { correctedQuery, explanation: "Fixed syntax error" },
    });

    const result = await cypherCorrectionNode(buildState());

    expect(result.originalQuery).toBe("MATCH (c:Course) RETURN c.name");
  });

  it("should pass schema to generateStructured", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { correctedQuery, explanation: "Fixed syntax error" },
    });

    await cypherCorrectionNode(buildState());

    const [systemPrompt] = mockLLMClient.generateStructured.mock.calls[0];
    expect(systemPrompt).toContain(mockSchema);
  });

  it("should return error when LLM fails", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: false,
      error: "LLM rate limit exceeded",
    });

    const result = await cypherCorrectionNode(buildState());

    expect(result.error).toBe(
      "Query correction failed: LLM rate limit exceeded",
    );
  });

  it("should return error when generateStructured throws", async () => {
    mockLLMClient.generateStructured.mockRejectedValue(
      new Error("Network error"),
    );

    const result = await cypherCorrectionNode(buildState());

    expect(result.error).toContain("Query correction failed");
    expect(result.error).toContain("Network error");
  });
});
