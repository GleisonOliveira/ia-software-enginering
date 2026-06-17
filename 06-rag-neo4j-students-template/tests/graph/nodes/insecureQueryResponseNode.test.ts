import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import type { OpenRouterService } from "../../../src/services/openrouterService.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import { createInsecureQueryResponseNode } from "../../../src/graph/nodes/insecureQueryResponseNode.ts";

function buildState(overrides?: Partial<GraphState>): GraphState {
  return {
    messages: [],
    ...overrides,
  } as GraphState;
}

describe("createInsecureQueryResponseNode", () => {
  let mockLLMClient: jest.Mocked<OpenRouterService>;
  let insecureQueryResponseNode: ReturnType<typeof createInsecureQueryResponseNode>;

  beforeEach(() => {
    mockLLMClient = {
      generateStructured: jest.fn(),
    } as unknown as jest.Mocked<OpenRouterService>;

    insecureQueryResponseNode = createInsecureQueryResponseNode(mockLLMClient);
  });

  it("should return analysis message when LLM responds successfully", async () => {
    const analysis =
      "The query contains dangerous commands and cannot be executed.";
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { analysis },
    });

    const result = await insecureQueryResponseNode(buildState());

    expect(result.messages).toBeDefined();
    expect(result.messages).toHaveLength(1);
    expect(result.messages![0].content).toBe(analysis);
  });

  it("should use the error from state as analysis context", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { analysis: "Friendly message for the user" },
    });

    await insecureQueryResponseNode(
      buildState({
        error: "The query MATCH (n) DETACH DELETE n is destructive",
      }),
    );

    const [, userPrompt] = mockLLMClient.generateStructured.mock.calls[0];
    expect(userPrompt).toContain("MATCH (n) DETACH DELETE n is destructive");
  });

  it("should pass fallback text when state has no error", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { analysis: "Friendly message" },
    });

    await insecureQueryResponseNode(buildState());

    const [, userPrompt] = mockLLMClient.generateStructured.mock.calls[0];
    expect(userPrompt).toContain("Não foi possível processar a busca");
  });

  it("should return fallback message when analysis is empty", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: { analysis: "" },
    });

    const result = await insecureQueryResponseNode(buildState());

    expect(result.messages).toBeDefined();
    expect(result.messages).toHaveLength(1);
    expect(result.messages![0].content).toBe("Não foi possível processar a busca");
  });

  it("should return fallback message when generateStructured fails", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: false,
      error: "LLM rate limit exceeded",
    });

    const result = await insecureQueryResponseNode(buildState());

    expect(result.messages).toBeDefined();
    expect(result.messages).toHaveLength(1);
    expect(result.messages![0].content).toBe("Não foi possível processar a busca");
  });

  it("should return fallback message when generateStructured throws", async () => {
    mockLLMClient.generateStructured.mockRejectedValue(
      new Error("Network error"),
    );

    const result = await insecureQueryResponseNode(buildState());

    expect(result.messages).toBeDefined();
    expect(result.messages).toHaveLength(1);
    expect(result.messages![0].content).toBe("Não foi possível processar a busca");
  });
});
