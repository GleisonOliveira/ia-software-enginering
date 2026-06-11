import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import type { OpenRouterService } from "../../../src/services/openrouterService.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import { createQueryPlannerNode } from "../../../src/graph/nodes/queryPlannerNode.ts";

function buildState(question: string, overrides?: Partial<GraphState>): GraphState {
  return {
    question,
    messages: [],
    ...overrides,
  } as GraphState;
}

describe("createQueryPlannerNode", () => {
  let mockLLMClient: jest.Mocked<OpenRouterService>;
  let queryPlannerNode: ReturnType<typeof createQueryPlannerNode>;

  beforeEach(() => {
    mockLLMClient = {
      generateStructured: jest.fn(),
    } as unknown as jest.Mocked<OpenRouterService>;

    queryPlannerNode = createQueryPlannerNode(mockLLMClient);
  });

  it("should mark query as simple when requiresDecomposition is false", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: {
        complexity: "simple",
        requiresDecomposition: false,
        subQuestions: [],
        reasoning: "Single entity, direct retrieval",
      },
    });

    const result = await queryPlannerNode(
      buildState("List all available courses"),
    );

    expect(result.isMultiStep).toBeUndefined();
    expect(result.subQuestions).toBeUndefined();
    expect(result.error).toBeUndefined();
  });

  it("should decompose query when requiresDecomposition is true with subQuestions", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: {
        complexity: "complex",
        requiresDecomposition: true,
        subQuestions: [
          "Average completion per course?",
          "Revenue for courses >70%?",
        ],
        reasoning: "Multiple aggregations needed",
      },
    });

    const result = await queryPlannerNode(
      buildState("Compare revenue between high vs low completion courses"),
    );

    expect(result.isMultiStep).toBe(true);
    expect(result.subQuestions).toEqual([
      "Average completion per course?",
      "Revenue for courses >70%?",
    ]);
    expect(result.currentStep).toBe(0);
    expect(result.subQueries).toEqual([]);
    expect(result.subResults).toEqual([]);
    expect(result.error).toBeUndefined();
  });

  it("should not decompose when requiresDecomposition is true but subQuestions is empty", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: {
        complexity: "complex",
        requiresDecomposition: true,
        subQuestions: [],
        reasoning: "No sub-questions generated",
      },
    });

    const result = await queryPlannerNode(
      buildState("Some complex query"),
    );

    expect(result.isMultiStep).toBeUndefined();
    expect(result.subQuestions).toBeUndefined();
  });

  it("should return error when generateStructured fails", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: false,
      error: "LLM rate limit exceeded",
    });

    const result = await queryPlannerNode(
      buildState("List all courses"),
    );

    expect(result.isMultiStep).toBe(false);
    expect(result.error).toBe("LLM rate limit exceeded");
  });

  it("should propagate existing state on success for simple queries", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: {
        complexity: "simple",
        requiresDecomposition: false,
        subQuestions: [],
        reasoning: "Simple query",
      },
    });

    const result = await queryPlannerNode(
      buildState("List all courses", { answer: "existing answer" }),
    );

    expect(result.answer).toBe("existing answer");
    expect(result.isMultiStep).toBeUndefined();
    expect(result.error).toBeUndefined();
  });

  it("should handle errors thrown by generateStructured", async () => {
    mockLLMClient.generateStructured.mockRejectedValue(
      new Error("Network error"),
    );

    const result = await queryPlannerNode(
      buildState("List all courses"),
    );

    expect(result.isMultiStep).toBe(false);
    expect(result.error).toContain("Failed to extract question");
    expect(result.error).toContain("Network error");
  });

  it("should pass system prompt, user prompt, and schema to generateStructured", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: {
        complexity: "simple",
        requiresDecomposition: false,
        subQuestions: [],
        reasoning: "Simple query",
      },
    });

    await queryPlannerNode(buildState("My test question"));

    expect(mockLLMClient.generateStructured.mock.calls).toHaveLength(1);
    const [systemPrompt, userPrompt] = mockLLMClient.generateStructured.mock.calls[0];

    expect(typeof systemPrompt).toBe("string");
    expect(userPrompt).toBe("My test question");
  });
});
