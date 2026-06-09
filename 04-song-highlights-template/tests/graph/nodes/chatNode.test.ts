import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import { AIMessage, HumanMessage } from "langchain";
import { createChatNode } from "../../../src/graph/nodes/chatNode.ts";
import type { OpenRouterService } from "../../../src/services/openrouterService.ts";
import type { PreferencesService } from "../../../src/services/preferencesService.ts";
import type { GraphState } from "../../../src/graph/graph.ts";

function createState(
  messages: Array<{ role: "user" | "ai"; content: string }>,
  overrides?: Partial<GraphState>,
): GraphState {
  return {
    messages: messages.map((msg) =>
      msg.role === "user"
        ? new HumanMessage(msg.content)
        : new AIMessage(msg.content),
    ),
    userId: "test-user",
    ...overrides,
  };
}

describe("createChatNode", () => {
  let mockLLMClient: jest.Mocked<OpenRouterService>;
  let mockPreferencesService: jest.Mocked<PreferencesService>;
  let chatNode: ReturnType<typeof createChatNode>;

  beforeEach(() => {
    mockLLMClient = {
      generateStructured: jest.fn(),
    } as unknown as jest.Mocked<OpenRouterService>;

    mockPreferencesService = {
      getBasicInfo: jest
        .fn<(userId: string) => Promise<string | undefined>>()
        .mockResolvedValue(undefined),
    } as unknown as jest.Mocked<PreferencesService>;

    chatNode = createChatNode(mockLLMClient, mockPreferencesService);
  });

  it("deve retornar mensagem de sucesso quando generateStructured retorna dados válidos com preferências", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: {
        message: "Que legal! Conte-me mais sobre suas músicas favoritas!",
        shouldSavePreferences: true,
        preferences: { name: "João", favoriteGenres: ["rock"] },
      },
    });

    const state = createState([
      { role: "user", content: "Oi! Meu nome é João e gosto de rock" },
    ]);

    const result = await chatNode(state);

    expect(result.messages).toHaveLength(1);
    const msg = result.messages![0];
    expect(msg).toBeInstanceOf(AIMessage);
    expect(msg.content).toBe(
      "Que legal! Conte-me mais sobre suas músicas favoritas!",
    );
    expect(result.extractedPreferences).toEqual({
      name: "João",
      favoriteGenres: ["rock"],
    });
    expect(result.needsSummarization).toBe(false);
  });

  it("deve retornar extractedPreferences como undefined quando shouldSavePreferences é false", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: {
        message: "Olá! Como posso ajudar você hoje?",
        shouldSavePreferences: false,
        preferences: null,
      },
    });

    const state = createState([{ role: "user", content: "Olá!" }]);

    const result = await chatNode(state);

    expect(result.messages).toHaveLength(1);
    const msg = result.messages![0];
    expect(msg).toBeInstanceOf(AIMessage);
    expect(msg.content).toBe("Olá! Como posso ajudar você hoje?");
    expect(result.extractedPreferences).toBeUndefined();
    expect(result.needsSummarization).toBe(false);
  });

  it("deve retornar mensagem de erro quando generateStructured retorna success: false", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: false,
      error: "Erro interno do LLM",
    });

    const state = createState([{ role: "user", content: "Olá!" }]);

    const result = await chatNode(state);

    expect(result.messages).toHaveLength(1);
    const msg = result.messages![0];
    expect(msg).toBeInstanceOf(AIMessage);
    expect(msg.content).toBe(
      "Desculpe encontrei um erro, por favor tente novamente",
    );
  });

  it("deve retornar mensagem de erro quando generateStructured retorna success: true sem data", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: undefined,
    });

    const state = createState([{ role: "user", content: "Olá!" }]);

    const result = await chatNode(state);

    expect(result.messages).toHaveLength(1);
    const msg = result.messages![0];
    expect(msg).toBeInstanceOf(AIMessage);
    expect(msg.content).toBe(
      "Desculpe encontrei um erro, por favor tente novamente",
    );
  });

  it("deve montar o histórico da conversa corretamente no user prompt", async () => {
    mockLLMClient.generateStructured.mockResolvedValue({
      success: true,
      data: {
        message: "Olá novamente!",
        shouldSavePreferences: false,
        preferences: null,
      },
    });

    const state = createState([
      { role: "user", content: "Olá!" },
      { role: "ai", content: "Oi! Como posso ajudar?" },
      { role: "user", content: "Gosto de rock" },
    ]);

    await chatNode(state);

    expect(mockLLMClient.generateStructured.mock.calls).toHaveLength(1);
    const userPrompt = mockLLMClient.generateStructured.mock.calls[0][1];
    const parsedPrompt = JSON.parse(userPrompt);
    expect(parsedPrompt.mensagem_atual_do_usuario).toBe("Gosto de rock");
    expect(parsedPrompt.contexto_da_conversa).toContain("User: Olá!");
    expect(parsedPrompt.contexto_da_conversa).toContain(
      "AI: Oi! Como posso ajudar?",
    );
    expect(parsedPrompt.contexto_da_conversa).toContain("User: Gosto de rock");
  });
});
