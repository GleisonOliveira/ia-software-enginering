import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import type { Neo4jService } from "../../../src/services/neo4jService.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import { createCypherExecutorNode } from "../../../src/graph/nodes/cypherExecutorNode.ts";

jest.mock("../../../src/config.ts", () => ({
  default: {
    maxCorrectionAttempts: 1,
  },
}));

function buildState(overrides?: Partial<GraphState>): GraphState {
  return {
    messages: [],
    query: "MATCH (c:Course) RETURN c.name",
    ...overrides,
  } as GraphState;
}

describe("createCypherExecutorNode", () => {
  let mockNeo4jService: jest.Mocked<Neo4jService>;
  let cypherExecutorNode: ReturnType<typeof createCypherExecutorNode>;

  beforeEach(() => {
    mockNeo4jService = {
      validateQuery: jest
        .fn<() => Promise<boolean>>()
        .mockResolvedValue(true),
      query: jest.fn<() => Promise<any[]>>(),
    } as unknown as jest.Mocked<Neo4jService>;

    cypherExecutorNode = createCypherExecutorNode(mockNeo4jService);
  });

  it("should return dbResults when query succeeds", async () => {
    const mockResults = [
      { name: "Course A", url: "https://example.com/a" },
    ];
    mockNeo4jService.query.mockResolvedValue(mockResults);

    const result = await cypherExecutorNode(buildState());

    expect(result.dbResults).toEqual(mockResults);
    expect(result.needsCorrection).toBe(false);
    expect(result.error).toBeUndefined();
  });

  it("should preserve other state properties on success", async () => {
    mockNeo4jService.query.mockResolvedValue([
      { name: "Course A" },
    ]);

    const result = await cypherExecutorNode(
      buildState({ question: "List courses", answer: "existing answer" }),
    );

    expect(result.dbResults).toBeDefined();
    expect(result.question).toBe("List courses");
    expect(result.answer).toBe("existing answer");
  });

  it("should pass query to neo4jService.query", async () => {
    mockNeo4jService.query.mockResolvedValue([
      { name: "Course A" },
    ]);

    await cypherExecutorNode(
      buildState({ query: "MATCH (c:Course) RETURN c.name" }),
    );

    expect(mockNeo4jService.query.mock.calls).toHaveLength(1);
    const [query] = mockNeo4jService.query.mock.calls[0];
    expect(query).toBe("MATCH (c:Course) RETURN c.name");
  });

  it("should return no results error when query returns empty array", async () => {
    mockNeo4jService.query.mockResolvedValue([]);

    const result = await cypherExecutorNode(buildState());

    expect(result.dbResults).toEqual([]);
    expect(result.error).toBe("No results found");
  });

  it("should return needsCorrection when query throws and attempts remain", async () => {
    mockNeo4jService.query.mockRejectedValue(
      new Error("Syntax error in query"),
    );

    const result = await cypherExecutorNode(
      buildState({ correctionAttempts: 0 }),
    );

    expect(result.needsCorrection).toBe(true);
    expect(result.validationError).toBeDefined();
    expect(result.validationError).toContain("Syntax error in query");
  });

  it("should return correction failed error when correction attempts exhausted", async () => {
    mockNeo4jService.query.mockRejectedValue(
      new Error("Syntax error in query"),
    );

    const result = await cypherExecutorNode(
      buildState({ correctionAttempts: 1 }),
    );

    expect(result.error).toBe("Invalid Cypher query - correction failed");
    expect(result.needsCorrection).toBeUndefined();
  });

  it("should progress subResults in multi-step mode", async () => {
    mockNeo4jService.query.mockResolvedValue([
      { name: "Course B" },
    ]);

    const result = await cypherExecutorNode(
      buildState({
        isMultiStep: true,
        subQuestions: ["List courses", "Count students"],
        currentStep: 0,
        subResults: [[{ name: "Course A" }]],
      }),
    );

    expect(result.dbResults).toEqual([{ name: "Course B" }]);
    expect(result.subResults).toEqual([
      [{ name: "Course A" }],
      { name: "Course B" },
    ]);
    expect(result.currentStep).toBe(1);
    expect(result.needsCorrection).toBe(false);
  });

  it("should initialize subResults in multi-step when none exist", async () => {
    mockNeo4jService.query.mockResolvedValue([
      { name: "Course A" },
    ]);

    const result = await cypherExecutorNode(
      buildState({
        isMultiStep: true,
        subQuestions: ["List courses"],
        currentStep: 0,
      }),
    );

    expect(result.dbResults).toEqual([{ name: "Course A" }]);
    expect(result.subResults).toEqual([{ name: "Course A" }]);
    expect(result.currentStep).toBe(1);
  });

  it("should not enter multi-step when isMultiStep is false", async () => {
    mockNeo4jService.query.mockResolvedValue([
      { name: "Course A" },
    ]);

    const result = await cypherExecutorNode(
      buildState({
        isMultiStep: false,
      }),
    );

    expect(result.dbResults).toEqual([{ name: "Course A" }]);
    expect(result.subResults).toBeUndefined();
    expect(result.currentStep).toBeUndefined();
    expect(result.needsCorrection).toBe(false);
  });

  it("should set originalQuery on correction path when query exists", async () => {
    mockNeo4jService.query.mockRejectedValue(
      new Error("Syntax error"),
    );

    const result = await cypherExecutorNode(
      buildState({
        query: "MATCH (c:Course) RETURN c.name",
        correctionAttempts: 0,
      }),
    );

    expect(result.originalQuery).toBe("MATCH (c:Course) RETURN c.name");
    expect(result.needsCorrection).toBe(true);
  });

  it("should use existing originalQuery on correction path", async () => {
    mockNeo4jService.query.mockRejectedValue(
      new Error("Syntax error"),
    );

    const result = await cypherExecutorNode(
      buildState({
        query: "MATCH (c:Course) RETURN c.name",
        originalQuery: "MATCH (c:Course) RETURN n",
        correctionAttempts: 0,
      }),
    );

    expect(result.originalQuery).toBe("MATCH (c:Course) RETURN n");
  });

  it("should handle outer catch when state is null", async () => {
    const result = await cypherExecutorNode(null as unknown as GraphState);

    expect(result.error).toBe("Invalid Cypher query - correction failed");
  });
});
