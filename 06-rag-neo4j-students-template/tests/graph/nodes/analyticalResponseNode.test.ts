import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import type { OpenRouterService } from "../../../src/services/openrouterService.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import { createAnalyticalResponseNode } from "../../../src/graph/nodes/analyticalResponseNode.ts";

function buildState(overrides?: Partial<GraphState>): GraphState {
  return {
    messages: [],
    question: "What is the sales total?",
    query: "MATCH (c:Course) RETURN c.name",
    ...overrides,
  } as GraphState;
}

describe("createAnalyticalResponseNode", () => {
  let mockLLMClient: jest.Mocked<OpenRouterService>;
  let analyticalResponseNode: ReturnType<typeof createAnalyticalResponseNode>;

  const mockAnswer = "The total revenue across all courses is $250,000.";
  const mockFollowUp = [
    "Which course has the highest revenue?",
  ];

  beforeEach(() => {
    mockLLMClient = {
      generateStructured: jest.fn(),
    } as unknown as jest.Mocked<OpenRouterService>;

    analyticalResponseNode = createAnalyticalResponseNode(mockLLMClient);
  });

  describe("error path", () => {
    it("should return LLM-generated error response when state.error is set", async () => {
      mockLLMClient.generateStructured.mockResolvedValue({
        success: true,
        data: { answer: mockAnswer, followUpQuestions: mockFollowUp },
      });

      const result = await analyticalResponseNode(
        buildState({ error: "Invalid query syntax" }),
      );

      expect(result.messages).toBeDefined();
      expect(result.messages).toHaveLength(1);
      expect(result.answer).toBe(mockAnswer);
      expect(result.followUpQuestions).toEqual(mockFollowUp);
    });

    it("should fallback when LLM fails on error path", async () => {
      mockLLMClient.generateStructured.mockResolvedValue({
        success: false,
        error: "LLM rate limit exceeded",
      });

      const result = await analyticalResponseNode(
        buildState({ error: "Invalid query syntax" }),
      );

      expect(result.messages).toBeDefined();
      expect(result.messages).toHaveLength(1);
      expect(result.answer).toContain("An error ocurred");
      expect(result.followUpQuestions).toEqual([]);
    });
  });

  describe("no results path", () => {
    it("should handle no results when dbResults is empty", async () => {
      mockLLMClient.generateStructured.mockResolvedValue({
        success: true,
        data: { answer: mockAnswer, followUpQuestions: mockFollowUp },
      });

      const result = await analyticalResponseNode(
        buildState({ dbResults: [] }),
      );

      expect(result.messages).toBeDefined();
      expect(result.messages).toHaveLength(1);
      expect(result.answer).toBe(mockAnswer);
      expect(result.followUpQuestions).toEqual(mockFollowUp);
    });

    it("should handle no results when dbResults is undefined", async () => {
      mockLLMClient.generateStructured.mockResolvedValue({
        success: true,
        data: { answer: mockAnswer, followUpQuestions: mockFollowUp },
      });

      const result = await analyticalResponseNode(buildState());

      expect(result.messages).toBeDefined();
      expect(result.messages).toHaveLength(1);
      expect(result.answer).toBe(mockAnswer);
    });

    it("should fallback when LLM fails on no results path", async () => {
      mockLLMClient.generateStructured.mockResolvedValue({
        success: false,
        error: "LLM error",
      });

      const result = await analyticalResponseNode(
        buildState({ dbResults: [] }),
      );

      expect(result.messages).toBeDefined();
      expect(result.messages).toHaveLength(1);
      expect(result.answer).toBe("No data found matching your query.");
      expect(result.followUpQuestions).toEqual([]);
    });
  });

  describe("success path", () => {
    it("should return analytical response when dbResults exist", async () => {
      mockLLMClient.generateStructured.mockResolvedValue({
        success: true,
        data: { answer: mockAnswer, followUpQuestions: mockFollowUp },
      });

      const result = await analyticalResponseNode(
        buildState({ dbResults: [{ name: "Course A", revenue: 50000 }] }),
      );

      expect(result.messages).toBeDefined();
      expect(result.messages).toHaveLength(1);
      expect(result.answer).toBe(mockAnswer);
      expect(result.followUpQuestions).toEqual(mockFollowUp);
    });

    it("should use multi-step synthesis when in multi-step mode", async () => {
      mockLLMClient.generateStructured.mockResolvedValue({
        success: true,
        data: { answer: mockAnswer, followUpQuestions: mockFollowUp },
      });

      await analyticalResponseNode(
        buildState({
          dbResults: [{ name: "Course A" }],
          isMultiStep: true,
          subResults: [[{ name: "Course A" }]],
          subQuestions: ["List courses"],
          subQueries: ["MATCH (c:Course) RETURN c.name"],
        }),
      );

      const [, userPrompt] =
        mockLLMClient.generateStructured.mock.calls[0];
      expect(userPrompt).toContain("original_question");
    });

    it("should not use multi-step synthesis when subResults is empty", async () => {
      mockLLMClient.generateStructured.mockResolvedValue({
        success: true,
        data: { answer: mockAnswer, followUpQuestions: mockFollowUp },
      });

      await analyticalResponseNode(
        buildState({
          dbResults: [{ name: "Course A" }],
          isMultiStep: true,
          subResults: [],
          subQuestions: ["List courses"],
          subQueries: ["MATCH (c:Course) RETURN c.name"],
        }),
      );

      const [, userPrompt] =
        mockLLMClient.generateStructured.mock.calls[0];
      expect(userPrompt).not.toContain("original_question");
      expect(userPrompt).toContain("dbResults");
    });

    it("should fallback when LLM fails on success path", async () => {
      mockLLMClient.generateStructured.mockResolvedValue({
        success: false,
        error: "LLM error",
      });

      const result = await analyticalResponseNode(
        buildState({ dbResults: [{ name: "Course A" }] }),
      );

      expect(result.error).toContain("Reponse generation faild");
    });
  });

  describe("exception handling", () => {
    it("should catch and return error when generateStructured throws", async () => {
      mockLLMClient.generateStructured.mockRejectedValue(
        new Error("Unexpected network failure"),
      );

      const result = await analyticalResponseNode(
        buildState({ dbResults: [{ name: "Course A" }] }),
      );

      expect(result.error).toBeDefined();
      expect(result.error).toContain("Response generation failed");
    });

    it("should preserve state when exception is thrown", async () => {
      mockLLMClient.generateStructured.mockRejectedValue(
        new Error("Network error"),
      );

      const result = await analyticalResponseNode(
        buildState({ question: "Sales report", dbResults: [{ name: "Course A" }] }),
      );

      expect(result.question).toBe("Sales report");
      expect(result.error).toContain("Response generation failed");
    });
  });
});
