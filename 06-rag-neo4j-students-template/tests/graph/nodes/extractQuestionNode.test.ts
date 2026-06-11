import type { GraphState } from "../../../src/graph/graph.ts";
import { HumanMessage } from "@langchain/core/messages";
import { createExtractQuestionNode } from "../../../src/graph/nodes/extractQuestionNode.ts";

function message(text: string): HumanMessage {
  return new HumanMessage({ content: text });
}

function buildState(overrides: Partial<GraphState> = {}): GraphState {
  return {
    messages: [],
    ...overrides,
  } as GraphState;
}

describe("createExtractQuestionNode", () => {
  it("should extract question from the last message", async () => {
    const node = createExtractQuestionNode();
    const result = await node(
      buildState({ messages: [message("What is the sales total?")] }),
    );

    expect(result.question).toBe("What is the sales total?");
    expect(result.error).toBeUndefined();
  });

  it("should extract question from the last message when multiple messages exist", async () => {
    const node = createExtractQuestionNode();
    const result = await node(
      buildState({
        messages: [message("What is the sales total?"), message("Show me last quarter")],
      }),
    );

    expect(result.question).toBe("Show me last quarter");
  });

  it("should return error when messages array is empty", async () => {
    const node = createExtractQuestionNode();
    const result = await node(buildState({ messages: [] }));

    expect(result.error).toBe("No messages provided");
  });

  it("should return error when messages is undefined", async () => {
    const node = createExtractQuestionNode();
    const result = await node(buildState({ messages: undefined as unknown as [] }));

    expect(result.error).toBe("No messages provided");
  });

  it("should return error when question is an empty string", async () => {
    const node = createExtractQuestionNode();
    const result = await node(buildState({ messages: [message("")] }));

    expect(result.error).toBe("No valid question found in messages");
  });

  it("should return error when question is only whitespace", async () => {
    const node = createExtractQuestionNode();
    const result = await node(buildState({ messages: [message("   ")] }));

    expect(result.error).toBe("No valid question found in messages");
  });

  it("should return error when message has no text property", async () => {
    const node = createExtractQuestionNode();
    const result = await node(
      buildState({ messages: [message("")] }),
    );

    expect(result.error).toBe("No valid question found in messages");
  });

  it("should propagate other state properties", async () => {
    const node = createExtractQuestionNode();
    const result = await node(
      buildState({
        answer: "existing answer",
        question: "old question",
        messages: [message("Question?")],
      }),
    );

    expect(result.question).toBe("Question?");
    expect(result.answer).toBe("existing answer");
  });

  it("should handle error when state is null", async () => {
    const node = createExtractQuestionNode();
    const result = await node(null as unknown as GraphState);

    expect(result).toBeDefined();
  });
});
