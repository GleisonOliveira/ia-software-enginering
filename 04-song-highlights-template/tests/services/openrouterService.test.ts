import { jest, describe, it, expect, beforeEach, afterEach } from "@jest/globals";
import { z } from "zod/v3";
import { AIMessageChunk } from "@langchain/core/messages";
import { BaseChatOpenAI, type BaseChatOpenAICallOptions } from "@langchain/openai";
import { OpenRouterService } from "../../src/services/openrouterService.ts";

const mockInvoke = jest.fn(
  async (..._args: Parameters<typeof BaseChatOpenAI.prototype.invoke>) =>
    new AIMessageChunk({ content: "" }),
);

function mockAIMessage(content: string): AIMessageChunk {
  return new AIMessageChunk({ content });
}

describe("OpenRouterService", () => {
  let service: OpenRouterService;

  beforeEach(() => {
    mockInvoke.mockReset();

    jest.spyOn(BaseChatOpenAI.prototype, "invoke").mockImplementation(mockInvoke);
    jest.spyOn(BaseChatOpenAI.prototype, "bindTools").mockImplementation(
      function (this: BaseChatOpenAI<BaseChatOpenAICallOptions>) {
        return this;
      },
    );
    jest.spyOn(BaseChatOpenAI.prototype, "profile", "get").mockReturnValue({ structuredOutput: true });

    service = new OpenRouterService();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe("generateStructured", () => {
    const testSchema = z.object({
      message: z.string(),
      score: z.number(),
    });

    it("deve retornar sucesso quando o modelo retorna dados válidos no schema", async () => {
      const validData = { message: "Tudo certo!", score: 42 };
      mockInvoke.mockResolvedValue(mockAIMessage(JSON.stringify(validData)));

      const result = await service.generateStructured(
        "Seja um assistente útil",
        "Qual é a resposta?",
        testSchema,
      );

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toEqual(validData);
      }
    });

    it("deve retornar erro quando o modelo retorna dados que não passam no parse do schema", async () => {
      mockInvoke.mockResolvedValue(mockAIMessage("invalid json content"));

      const result = await service.generateStructured(
        "Seja um assistente útil",
        "Qual é a resposta?",
        testSchema,
      );

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error).toContain("providerStrategy");
    });

    it("deve retornar erro quando os dados retornados não correspondem ao schema", async () => {
      const typeMismatchData = { message: 123, score: "nao é numero" };
      mockInvoke.mockResolvedValue(mockAIMessage(JSON.stringify(typeMismatchData)));

      const result = await service.generateStructured(
        "Seja um assistente útil",
        "Qual é a resposta?",
        testSchema,
      );

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it("deve retornar erro quando o modelo lança uma exceção do tipo Error", async () => {
      mockInvoke.mockRejectedValue(new Error("Erro de conexão com a API"));

      const result = await service.generateStructured(
        "Seja um assistente útil",
        "Qual é a resposta?",
        testSchema,
      );

      expect(result.success).toBe(false);
      expect(result.error).toBe("Erro de conexão com a API");
    });

    it("deve retornar erro quando o modelo lança uma exceção que não é instância de Error", async () => {
      mockInvoke.mockImplementation(() => Promise.reject("Erro de string simples"));

      const result = await service.generateStructured(
        "Seja um assistente útil",
        "Qual é a resposta?",
        testSchema,
      );

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it("deve retornar erro quando o schema tem campos obrigatórios faltando", async () => {
      const invalidData = { message: "incompleto" };
      mockInvoke.mockResolvedValue(mockAIMessage(JSON.stringify(invalidData)));

      const result = await service.generateStructured(
        "Seja um assistente útil",
        "Qual é a resposta?",
        testSchema,
      );

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });
  });
});
