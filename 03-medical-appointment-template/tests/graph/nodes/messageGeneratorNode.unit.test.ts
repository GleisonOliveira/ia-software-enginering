jest.mock("langchain", () => ({
  AIMessage: class {
    content: string;
    constructor(content: string) {
      this.content = content;
    }
  },
}));

import { describe, it, expect, jest } from "@jest/globals";
import { createMessageGeneratorNode } from "../../../src/graph/nodes/messageGeneratorNode.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import type { MessageResponse } from "../../../src/prompts/v1/messageGenerator.ts";
import type { OpenRouterService } from "../../../src/services/openRouterService.ts";
import { HumanMessage } from "@langchain/core/messages";
import type { z } from "zod/v3";

type MockGenerateStructured = (
  systemPrompt: string,
  userPrompt: string,
  schema: z.ZodType<MessageResponse>,
) => Promise<
  | { success: true; data: MessageResponse }
  | { success: false; error: string }
>;

describe("createMessageGeneratorNode", () => {
  const baseMessage = new HumanMessage("Quero agendar uma consulta");

  it("should append generated message on success", async () => {
    const mockData: MessageResponse = {
      message: "Sua consulta foi agendada com sucesso!",
    };

    const mockLLM = {
      generateStructured: jest
        .fn<MockGenerateStructured>()
        .mockResolvedValue({ success: true, data: mockData }),
    } as unknown as OpenRouterService;

    const node = createMessageGeneratorNode(mockLLM);
    const state: GraphState = {
      messages: [baseMessage],
      intent: "schedule",
      actionSuccess: true,
      professionalName: "Dr. Test",
      datetime: "2026-06-02T10:00:00.000Z",
      patientName: "Maria",
    };
    const result = await node(state);

    expect(result.messages).toHaveLength(2);
    expect(result.messages![1]).toHaveProperty("content", mockData.message);
    expect(mockLLM.generateStructured).toHaveBeenCalledTimes(1);
  });

  it("should return fallback message when generateStructured returns error", async () => {
    const mockLLM = {
      generateStructured: jest
        .fn<MockGenerateStructured>()
        .mockResolvedValue({
          success: false,
          error: "Failed to generate message",
        }),
    } as unknown as OpenRouterService;

    const node = createMessageGeneratorNode(mockLLM);
    const state: GraphState = {
      messages: [baseMessage],
      intent: "schedule",
      actionSuccess: false,
    };
    const result = await node(state);

    expect(result.messages).toHaveLength(2);
    expect(result.messages![1]).toHaveProperty("content", "Desculpe, errei.");
    expect(mockLLM.generateStructured).toHaveBeenCalledTimes(1);
  });

  it("should return fallback message when generateStructured returns no data", async () => {
    const mockLLM = {
      generateStructured: jest
        .fn<MockGenerateStructured>()
        .mockResolvedValue({ success: true, data: undefined as any }),
    } as unknown as OpenRouterService;

    const node = createMessageGeneratorNode(mockLLM);
    const state: GraphState = {
      messages: [baseMessage],
      intent: "unknown",
    };
    const result = await node(state);

    expect(result.messages).toHaveLength(2);
    expect(result.messages![1]).toHaveProperty("content", "Desculpe, errei.");
    expect(mockLLM.generateStructured).toHaveBeenCalledTimes(1);
  });

  it("should return error message when generateStructured throws", async () => {
    const mockLLM = {
      generateStructured: jest
        .fn<MockGenerateStructured>()
        .mockRejectedValue(new Error("API timeout")),
    } as unknown as OpenRouterService;

    const node = createMessageGeneratorNode(mockLLM);
    const state: GraphState = {
      messages: [baseMessage],
      intent: "cancel",
      actionSuccess: true,
    };
    const result = await node(state);

    expect(result.messages).toHaveLength(2);
    expect(result.messages![1]).toHaveProperty(
      "content",
      "An error occurred while processing your request.",
    );
    expect(mockLLM.generateStructured).toHaveBeenCalledTimes(1);
  });

  it("should handle non-Error thrown values", async () => {
    const mockLLM = {
      generateStructured: jest
        .fn<MockGenerateStructured>()
        .mockRejectedValue("string error"),
    } as unknown as OpenRouterService;

    const node = createMessageGeneratorNode(mockLLM);
    const state: GraphState = {
      messages: [baseMessage],
    };
    const result = await node(state);

    expect(result.messages).toHaveLength(2);
    expect(result.messages![1]).toHaveProperty(
      "content",
      "An error occurred while processing your request.",
    );
  });
});
