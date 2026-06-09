import { jest, describe, it, expect, beforeEach, afterEach } from "@jest/globals";
import type { Runtime } from "@langchain/langgraph";
import { AIMessage, HumanMessage } from "langchain";
import { RemoveMessage } from "@langchain/core/messages";
import { createSummarizationNode } from "../../../src/graph/nodes/summarizationNode.ts";
import type { OpenRouterService } from "../../../src/services/openrouterService.ts";
import type { PreferencesService } from "../../../src/services/preferencesService.ts";
import type { GraphState } from "../../../src/graph/graph.ts";

function createMessage(role: "user" | "ai", content: string, id: string) {
  const msg =
    role === "user"
      ? new HumanMessage({ content, id })
      : new AIMessage({ content, id });
  return msg;
}

function createState(
  messages: Array<{ role: "user" | "ai"; content: string; id: string }>,
  overrides?: Partial<GraphState>,
): GraphState {
  return {
    messages: messages.map((m) => createMessage(m.role, m.content, m.id)),
    ...overrides,
  };
}

describe("createSummarizationNode", () => {
  let mockGenerateStructured: jest.Mock<
    (systemPrompt: string, userPrompt: string, schema: unknown) => Promise<{
      success: boolean;
      data?: unknown;
      error?: string;
    }>
  >;
  let mockStoreSummary: jest.Mock<
    (userId: string, summary: unknown) => Promise<void>
  >;
  let mockLLMClient: jest.Mocked<OpenRouterService>;
  let mockPreferencesService: jest.Mocked<PreferencesService>;
  let summarizationNode: ReturnType<typeof createSummarizationNode>;

  beforeEach(() => {
    mockGenerateStructured = jest.fn<
      (
        systemPrompt: string,
        userPrompt: string,
        schema: unknown,
      ) => Promise<{
        success: boolean;
        data?: unknown;
        error?: string;
      }>
    >();
    mockStoreSummary = jest.fn<
      (userId: string, summary: unknown) => Promise<void>
    >();

    mockLLMClient = {
      generateStructured: mockGenerateStructured,
    } as unknown as jest.Mocked<OpenRouterService>;

    mockPreferencesService = {
      storeSummary: mockStoreSummary,
    } as unknown as jest.Mocked<PreferencesService>;

    summarizationNode = createSummarizationNode(
      mockLLMClient,
      mockPreferencesService,
    );
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("deve retornar needsSummarization false quando generateStructured retorna erro", async () => {
    mockGenerateStructured.mockResolvedValue({
      success: false,
      error: "erro no llm",
    });

    const state = createState([
      { role: "user", content: "Oi", id: "1" },
      { role: "ai", content: "Olá!", id: "2" },
    ]);
    const runtime = { context: { userId: "user-1" } } as unknown as Runtime;

    const result = await summarizationNode(state, runtime);

    expect(result.needsSummarization).toBe(false);
    expect(mockStoreSummary).not.toHaveBeenCalled();
  });

  it("deve retornar needsSummarization false quando generateStructured retorna success sem data", async () => {
    mockGenerateStructured.mockResolvedValue({
      success: true,
      data: undefined,
    });

    const state = createState([
      { role: "user", content: "Oi", id: "1" },
      { role: "ai", content: "Olá!", id: "2" },
    ]);
    const runtime = { context: { userId: "user-1" } } as unknown as Runtime;

    const result = await summarizationNode(state, runtime);

    expect(result.needsSummarization).toBe(false);
    expect(mockStoreSummary).not.toHaveBeenCalled();
  });

  it("deve salvar o sumário e retornar dados corretos quando a geração é bem-sucedida", async () => {
    const summaryData = {
      name: "João",
      keyPreferences: "Gosta de rock e metal",
    };

    mockGenerateStructured.mockResolvedValue({
      success: true,
      data: summaryData,
    });

    const state = createState([
      { role: "user", content: "Oi", id: "1" },
      { role: "ai", content: "Olá!", id: "2" },
      { role: "user", content: "Gosto de rock", id: "3" },
    ]);
    const runtime = { context: { userId: "user-42" } } as unknown as Runtime;

    const result = await summarizationNode(state, runtime);

    expect(mockStoreSummary).toHaveBeenCalledWith("user-42", summaryData);
    expect(result.conversationSummary).toEqual(summaryData);
    expect(result.needsSummarization).toBe(false);
  });

  it("deve deletar as primeiras 2 mensagens quando o sumário é gerado", async () => {
    const summaryData = { keyPreferences: "Gosta de rock" };

    mockGenerateStructured.mockResolvedValue({
      success: true,
      data: summaryData,
    });

    const state = createState([
      { role: "user", content: "Oi", id: "msg-1" },
      { role: "ai", content: "Olá!", id: "msg-2" },
      { role: "user", content: "Gosto de rock", id: "msg-3" },
    ]);
    const runtime = { context: { userId: "user-1" } } as unknown as Runtime;

    const result = await summarizationNode(state, runtime);

    expect(result.messages).toHaveLength(2);
    expect(result.messages![0]).toBeInstanceOf(RemoveMessage);
    expect((result.messages![0] as RemoveMessage).id).toBe("msg-1");
    expect(result.messages![1]).toBeInstanceOf(RemoveMessage);
    expect((result.messages![1] as RemoveMessage).id).toBe("msg-2");
  });

  it("deve usar userId do state quando runtime não tem userId", async () => {
    const summaryData = { keyPreferences: "Gosta de pop" };

    mockGenerateStructured.mockResolvedValue({
      success: true,
      data: summaryData,
    });

    const state = createState(
      [
        { role: "user", content: "Oi", id: "1" },
        { role: "ai", content: "Oi!", id: "2" },
      ],
      { userId: "user-state" },
    );
    const runtime = { context: {} } as unknown as Runtime;

    await summarizationNode(state, runtime);

    expect(mockStoreSummary).toHaveBeenCalledWith("user-state", summaryData);
  });

  it('deve usar fallback "unknowm" quando nenhum userId é fornecido', async () => {
    const summaryData = { keyPreferences: "Gosta de mpb" };

    mockGenerateStructured.mockResolvedValue({
      success: true,
      data: summaryData,
    });

    const state = createState([
      { role: "user", content: "Oi", id: "1" },
      { role: "ai", content: "Oi!", id: "2" },
    ]);

    await summarizationNode(state, {} as unknown as Runtime);

    expect(mockStoreSummary).toHaveBeenCalledWith("unknowm", summaryData);
  });

  it("deve incluir conversationSummary anterior no prompt quando existir", async () => {
    const previousSummary = {
      keyPreferences: "Gosta de rock",
    };
    const newSummary = {
      keyPreferences: "Gosta de rock e mpb",
    };

    mockGenerateStructured.mockResolvedValue({
      success: true,
      data: newSummary,
    });

    const state = createState(
      [
        { role: "user", content: "Oi", id: "1" },
        { role: "ai", content: "Oi!", id: "2" },
      ],
      { conversationSummary: previousSummary, userId: "user-1" },
    );
    const runtime = { context: { userId: "user-1" } } as unknown as Runtime;

    const result = await summarizationNode(state, runtime);

    expect(result.conversationSummary).toEqual(newSummary);
    expect(mockStoreSummary).toHaveBeenCalledWith("user-1", newSummary);
  });

  it("deve propagar erro quando storeSummary falha", async () => {
    const summaryData = { keyPreferences: "Gosta de rock" };

    mockGenerateStructured.mockResolvedValue({
      success: true,
      data: summaryData,
    });
    mockStoreSummary.mockRejectedValue(new Error("store error"));

    const state = createState(
      [
        { role: "user", content: "Oi", id: "1" },
        { role: "ai", content: "Oi!", id: "2" },
      ],
      { userId: "user-1" },
    );
    const runtime = { context: { userId: "user-1" } } as unknown as Runtime;

    await expect(summarizationNode(state, runtime)).rejects.toThrow(
      "store error",
    );
  });
});
