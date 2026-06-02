import { describe, it, expect, jest } from "@jest/globals";
import { createIdentifyIntentNode } from "../../../src/graph/nodes/identifyIntentNode.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import type { IntentData } from "../../../src/prompts/v1/identifyIntent.ts";
import type { OpenRouterService } from "../../../src/services/openRouterService.ts";
import { HumanMessage } from "@langchain/core/messages";
import type { z } from "zod/v3";

type MockGenerateStructured = (
  systemPrompt: string,
  userPrompt: string,
  schema: z.ZodType<IntentData>,
) => Promise<
  | { success: true; data: IntentData }
  | { success: false; error: string }
>;

describe("createIdentifyIntentNode", () => {
  const baseMessage = new HumanMessage("Quero agendar uma consulta");

  it("should spread extracted data on success", async () => {
    const mockData: IntentData = {
      intent: "schedule",
      professionalId: 1,
      professionalName: "Dr. Test",
      datetime: "2026-06-02T10:00:00.000Z",
      patientName: "Maria",
    };

    const mockLLM = {
      generateStructured: jest
        .fn<MockGenerateStructured>()
        .mockResolvedValue({ success: true, data: mockData }),
    } as unknown as OpenRouterService;

    const node = createIdentifyIntentNode(mockLLM);
    const state: GraphState = { messages: [baseMessage] };
    const result = await node(state);

    expect(result.intent).toBe("schedule");
    expect(result.professionalId).toBe(1);
    expect(result.patientName).toBe("Maria");
    expect(result.error).toBeUndefined();
    expect(mockLLM.generateStructured).toHaveBeenCalledTimes(1);
  });

  it("should return unknown intent when generateStructured fails", async () => {
    const mockLLM = {
      generateStructured: jest
        .fn<MockGenerateStructured>()
        .mockResolvedValue({
          success: false,
          error: "Failed to identify intent",
        }),
    } as unknown as OpenRouterService;

    const node = createIdentifyIntentNode(mockLLM);
    const state: GraphState = { messages: [baseMessage] };
    const result = await node(state);

    expect(result.intent).toBe("unknown");
    expect(result.error).toBe("Failed to identify intent");
    expect(mockLLM.generateStructured).toHaveBeenCalledTimes(1);
  });

  it("should spread state and set unknown intent when generateStructured throws", async () => {
    const mockLLM = {
      generateStructured: jest
        .fn<MockGenerateStructured>()
        .mockRejectedValue(new Error("API timeout")),
    } as unknown as OpenRouterService;

    const state: GraphState = {
      messages: [baseMessage],
      patientName: "Joao",
    };
    const node = createIdentifyIntentNode(mockLLM);
    const result = await node(state);

    expect(result.intent).toBe("unknown");
    expect(result.error).toBe("API timeout");
    expect(result.patientName).toBe("Joao");
    expect(mockLLM.generateStructured).toHaveBeenCalledTimes(1);
  });

  it("should handle non-Error thrown values", async () => {
    const mockLLM = {
      generateStructured: jest
        .fn<MockGenerateStructured>()
        .mockRejectedValue("string error"),
    } as unknown as OpenRouterService;

    const node = createIdentifyIntentNode(mockLLM);
    const state: GraphState = { messages: [baseMessage] };
    const result = await node(state);

    expect(result.intent).toBe("unknown");
    expect(result.error).toBe("Intent identification failed");
  });
});
