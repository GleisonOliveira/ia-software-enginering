import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import type { OpenRouterService } from "../../../src/services/openrouterService.ts";
import type { Neo4jService } from "../../../src/services/neo4jService.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import { createCypherGeneratorNode } from "../../../src/graph/nodes/cypherGeneratorNode.ts";

function buildState(overrides?: Partial<GraphState>): GraphState {
  return {
    question: "List all courses",
    messages: [],
    ...overrides,
  } as GraphState;
}

describe("createCypherGeneratorNode", () => {
  let mockLLMClient: jest.Mocked<OpenRouterService>;
  let mockNeo4jService: jest.Mocked<Neo4jService>;
  let cypherGeneratorNode: ReturnType<typeof createCypherGeneratorNode>;

  const mockSchema = "Node properties: Course {name: string, url: string}";

  beforeEach(() => {
    mockLLMClient = {
      generateStructured: jest.fn(),
    } as unknown as jest.Mocked<OpenRouterService>;

    mockNeo4jService = {
      getSchema: jest.fn<() => Promise<string>>().mockResolvedValue(mockSchema),
    } as unknown as jest.Mocked<Neo4jService>;

    cypherGeneratorNode = createCypherGeneratorNode(
      mockLLMClient,
      mockNeo4jService,
    );
  });

  it("should generate a Cypher query for a simple question", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { query: "MATCH (c:Course) RETURN c.name AS courseName" },
    });

    const result = await cypherGeneratorNode(buildState());

    expect(result.query).toBe(
      "MATCH (c:Course) RETURN c.name AS courseName",
    );
    expect(result.error).toBeUndefined();
  });

  it("should pass schema to generateStructured", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { query: "MATCH (c:Course) RETURN c.name AS courseName" },
    });

    await cypherGeneratorNode(buildState());

    expect(mockNeo4jService.getSchema).toHaveBeenCalledTimes(1);
    expect(mockLLMClient.generateStructured.mock.calls).toHaveLength(1);
    const [systemPrompt] = mockLLMClient.generateStructured.mock.calls[0];
    expect(systemPrompt).toContain(mockSchema);
  });

  it("should use state.question as user prompt for non-multistep queries", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { query: "MATCH (c:Course) RETURN c.name AS courseName" },
    });

    await cypherGeneratorNode(buildState({ question: "List all courses" }));

    const [, userPrompt] = mockLLMClient.generateStructured.mock.calls[0];
    expect(userPrompt).toBe("List all courses");
  });

  it("should return error when generateStructured fails", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: false,
      error: "LLM rate limit exceeded",
    });

    const result = await cypherGeneratorNode(buildState());

    expect(result.error).toBe(
      "Failed to generate query: LLM rate limit exceeded",
    );
    expect(result.query).toBeUndefined();
  });

  it("should handle errors thrown by generateStructured", async () => {
    mockLLMClient.generateStructured.mockRejectedValue(
      new Error("Network error"),
    );

    const result = await cypherGeneratorNode(buildState());

    expect(result.error).toContain("Failed to generate query");
    expect(result.error).toContain("Network error");
  });

  it("should propagate existing state on error thrown", async () => {
    mockLLMClient.generateStructured.mockRejectedValue(
      new Error("Network error"),
    );

    const result = await cypherGeneratorNode(
      buildState({ answer: "existing answer" }),
    );

    expect(result.answer).toBe("existing answer");
    expect(result.error).toContain("Failed to generate query");
  });

  it("should append query to subQueries in multi-step mode", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { query: "MATCH (c:Course) RETURN c.name" },
    });

    const result = await cypherGeneratorNode(
      buildState({
        question: "Compare revenue between courses",
        isMultiStep: true,
        subQuestions: [
          "Average completion per course?",
          "Revenue for courses >70%?",
        ],
        currentStep: 0,
        subQueries: ["MATCH (s:Student)-[p:PURCHASED]->(c:Course) RETURN c.name"],
      }),
    );

    expect(result.query).toBe("MATCH (c:Course) RETURN c.name");
    expect(result.subQueries).toEqual([
      "MATCH (s:Student)-[p:PURCHASED]->(c:Course) RETURN c.name",
      "MATCH (c:Course) RETURN c.name",
    ]);
  });

  it("should not append to subQueries on the first multi-step call when subQueries is empty", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { query: "MATCH (c:Course) RETURN c.name" },
    });

    const result = await cypherGeneratorNode(
      buildState({
        question: "Compare revenue between courses",
        isMultiStep: true,
        subQuestions: [
          "Average completion per course?",
          "Revenue for courses >70%?",
        ],
        currentStep: 0,
        subQueries: [],
      }),
    );

    expect(result.query).toBe("MATCH (c:Course) RETURN c.name");
    expect(result.subQueries).toBeUndefined();
  });

  it("should use subQuestion for user prompt when in multi-step mode", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { query: "MATCH (c:Course) RETURN c.name" },
    });

    await cypherGeneratorNode(
      buildState({
        question: "Compare revenue between courses",
        isMultiStep: true,
        subQuestions: ["Average completion per course?"],
        currentStep: 0,
        subQueries: [],
      }),
    );

    const [, userPrompt] = mockLLMClient.generateStructured.mock.calls[0];
    expect(userPrompt).toBe("Average completion per course?");
  });

  it("should fall back to state.question when step question is out of bounds", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { query: "MATCH (c:Course) RETURN c.name" },
    });

    await cypherGeneratorNode(
      buildState({
        question: "Fallback question",
        isMultiStep: true,
        subQuestions: ["Sub question"],
        currentStep: 5,
        subQueries: [],
      }),
    );

    const [, userPrompt] = mockLLMClient.generateStructured.mock.calls[0];
    expect(userPrompt).toBe("Fallback question");
  });

  it("should not return subQueries when not in multi-step mode even if subQueries exist", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { query: "MATCH (c:Course) RETURN c.name" },
    });

    const result = await cypherGeneratorNode(
      buildState({
        isMultiStep: false,
        subQueries: ["some existing query"],
      }),
    );

    expect(result.query).toBe("MATCH (c:Course) RETURN c.name");
    expect(result.subQueries).toBeUndefined();
  });
});
