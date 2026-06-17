import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import type { OpenRouterService } from "../../../src/services/openrouterService.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import { createCypherValidatorNode } from "../../../src/graph/nodes/cypherValidatorNode.ts";

function buildState(overrides?: Partial<GraphState>): GraphState {
  return {
    messages: [],
    query: "MATCH (c:Course) RETURN c.name",
    ...overrides,
  } as GraphState;
}

describe("createCypherValidatorNode", () => {
  let mockLLMClient: jest.Mocked<OpenRouterService>;
  let cypherValidatorNode: ReturnType<typeof createCypherValidatorNode>;

  beforeEach(() => {
    mockLLMClient = {
      generateStructured: jest.fn(),
    } as unknown as jest.Mocked<OpenRouterService>;

    cypherValidatorNode = createCypherValidatorNode(mockLLMClient);
  });

  it("should return secure: true when query is empty", async () => {
    const result = await cypherValidatorNode(buildState({ query: "" }));

    expect(result.secure).toBe(true);
  });

  it("should return secure: true when query is undefined", async () => {
    const result = await cypherValidatorNode(buildState({ query: undefined }));

    expect(result.secure).toBe(true);
  });

  it("should return secure: true when LLM validates as safe", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { secure: true, analysis: "Query is safe" },
    });

    const result = await cypherValidatorNode(buildState());

    expect(result.secure).toBe(true);
    expect(result.error).toBeUndefined();
  });

  it("should return secure: false when LLM marks as unsafe", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: {
        secure: false,
        analysis: "The query contains destructive commands",
      },
    });

    const result = await cypherValidatorNode(buildState());

    expect(result.secure).toBe(false);
    expect(result.error).toBe("The query contains destructive commands");
  });

  it("should return secure: false when LLM call fails", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: false,
      error: "LLM error",
    });

    const result = await cypherValidatorNode(buildState());

    expect(result.secure).toBe(false);
    expect(result.error).toBe("Query can not be validated");
  });

  it("should return error when generateStructured throws", async () => {
    mockLLMClient.generateStructured.mockRejectedValue(
      new Error("Network error"),
    );

    const result = await cypherValidatorNode(buildState());

    expect(result.error).toBe("Query can not be validated");
    expect(result.secure).toBeUndefined();
  });
});
