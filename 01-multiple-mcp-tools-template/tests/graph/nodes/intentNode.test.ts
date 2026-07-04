import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import type { OpenRouterService } from "../../../src/services/openRouterService.ts";
import type { GraphState } from "../../../src/graph/state.ts";
import { intentNode } from "../../../src/graph/nodes/intentNode.ts";
import { HumanMessage } from "@langchain/core/messages";
import type { IntentData } from "../../../src/prompts/v1/identifyIntent.ts";

function buildState(overrides?: Partial<GraphState>): GraphState {
  return {
    messages: [new HumanMessage("list the top 5 products")],
    ...overrides,
  } as unknown as GraphState;
}

describe("intentNode", () => {
  let mockLLMClient: jest.Mocked<OpenRouterService>;
  let node: ReturnType<typeof intentNode>;

  const validIntentData: IntentData = {
    intent: "List the top 5 most sold products",
    fileContent: "product_id,product_name,units_sold\n1,Widget A,100\n2,Widget B,80",
    fileName: "sales_report",
    fileType: "csv",
  };

  beforeEach(() => {
    mockLLMClient = {
      generateStructured: jest.fn(),
    } as unknown as jest.Mocked<OpenRouterService>;

    node = intentNode(mockLLMClient);
  });

  it("should extract intent from user message", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      data: validIntentData,
    });

    const result = await node(buildState());

    expect(result.intent).toBe("List the top 5 most sold products");
    expect(result.error).toBeUndefined();
  });

  it("should extract file metadata from user message", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      data: validIntentData,
    });

    const result = await node(buildState());

    expect(result.fileContent).toBe(
      "product_id,product_name,units_sold\n1,Widget A,100\n2,Widget B,80",
    );
    expect(result.fileName).toBe("sales_report");
  });

  it("should not set error on happy path", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      data: validIntentData,
    });

    const result = await node(buildState());

    expect(result.error).toBeUndefined();
  });

  it("should handle missing messages gracefully and return error", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      data: validIntentData,
    });

    const result = await node(buildState({ messages: [] }));

    expect(result.error).toBeDefined();
    expect(result.messages).toBeDefined();
  });

  it("should return error when data is invalid", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      data: undefined,
    });

    const result = await node(buildState());

    expect(result.error).toBeDefined();
    expect(result.messages).toHaveLength(1);
    expect(result.messages![0].text).toContain("Sorry");
  });

  it("should propagate error from generateStructured rejection", async () => {
    mockLLMClient.generateStructured.mockRejectedValue(
      new Error("LLM rate limit exceeded"),
    );

    const result = await node(buildState());

    expect(result.error).toBe("LLM rate limit exceeded");
    expect(result.messages).toHaveLength(1);
    expect(result.messages![0].text).toContain("Sorry");
  });

  it("should set default fileName when fileName is null", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      data: { ...validIntentData, fileName: null },
    });

    const result = await node(buildState());

    expect(result.fileName).toBe("data.csv");
  });
});
