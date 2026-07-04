import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import type { OpenRouterService } from "../../../src/services/openRouterService.ts";
import type { GraphState } from "../../../src/graph/state.ts";
import { agentNode } from "../../../src/graph/nodes/agentNode.ts";
import { HumanMessage } from "@langchain/core/messages";

function buildState(overrides?: Partial<GraphState>): GraphState {
  return {
    messages: [new HumanMessage("list the top 5 products")],
    intent: "list the top 5 most sold products",
    fileName: "sales.csv",
    fileContent: "product_id,product_name,units_sold\n1,Widget A,100\n2,Widget B,80",
    ...overrides,
  } as unknown as GraphState;
}

describe("agentNode", () => {
  let mockLLMClient: jest.Mocked<OpenRouterService>;
  let node: ReturnType<typeof agentNode>;

  beforeEach(() => {
    mockLLMClient = {
      generateStructured: jest.fn(),
    } as unknown as jest.Mocked<OpenRouterService>;

    node = agentNode(mockLLMClient);
  });

  it("should return agent response message", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      data: "Top 5 products: Widget A, Widget B, Widget C, Widget D, Widget E",
    });

    const result = await node(buildState());

    expect(result.messages).toHaveLength(1);
    expect(result.messages![0].text).toBe(
      "Top 5 products: Widget A, Widget B, Widget C, Widget D, Widget E",
    );
  });

  it("should not set error on happy path", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      data: "Top 5 products: Widget A, Widget B",
    });

    const result = await node(buildState());

    expect(result.error).toBeUndefined();
  });

  it("should pass intent and file context to the LLM", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      data: "processed result",
    });

    await node(buildState());

    expect(mockLLMClient.generateStructured).toHaveBeenCalledTimes(1);
    const [systemPrompt, userMessage] =
      mockLLMClient.generateStructured.mock.calls[0];
    expect(systemPrompt).toContain("data processing agent");
    expect(userMessage).toContain("list the top 5 most sold products");
    expect(userMessage).toContain("sales.csv");
    expect(userMessage).toContain("Widget A");
  });

  it("should handle LLM failure gracefully", async () => {
    mockLLMClient.generateStructured.mockRejectedValue(
      new Error("LLM rate limit exceeded"),
    );

    const result = await node(buildState());

    expect(result.error).toBe("LLM rate limit exceeded");
    expect(result.messages).toHaveLength(1);
    expect(result.messages![0].text).toContain("Sorry");
  });

  it("should include state values in the LLM prompt even when empty", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      data: "result",
    });

    await node(buildState({
      intent: "",
      fileName: "data.csv",
      fileContent: "",
    }));

    const [, userMessage] = mockLLMClient.generateStructured.mock.calls[0];
    expect(userMessage).toContain("data.csv");
  });
});
