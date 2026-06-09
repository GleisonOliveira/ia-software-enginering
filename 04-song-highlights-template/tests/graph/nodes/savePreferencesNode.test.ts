import { jest, describe, it, expect, beforeEach } from "@jest/globals";
import type { Runtime } from "@langchain/langgraph";
import { createSavePreferencesNode } from "../../../src/graph/nodes/savePreferencesNode.ts";
import type { PreferencesService } from "../../../src/services/preferencesService.ts";
import type { GraphState } from "../../../src/graph/graph.ts";
import type { UserPreferences } from "../../../src/prompts/v1/chatResponse.ts";

describe("createSavePreferencesNode", () => {
  let mockMergePreferences: jest.Mock<
    (userId: string, prefs: UserPreferences) => Promise<void>
  >;
  let mockPreferencesService: jest.Mocked<PreferencesService>;
  let savePreferencesNode: ReturnType<typeof createSavePreferencesNode>;

  beforeEach(() => {
    mockMergePreferences = jest.fn();
    mockPreferencesService = {
      mergePreferences: mockMergePreferences,
    } as unknown as jest.Mocked<PreferencesService>;

    savePreferencesNode = createSavePreferencesNode(mockPreferencesService);
  });

  it("deve retornar objeto vazio quando extractedPreferences é undefined", async () => {
    const state: GraphState = { messages: [] };

    const result = await savePreferencesNode(state);

    expect(result).toEqual({});
    expect(mockMergePreferences).not.toHaveBeenCalled();
  });

  it("deve usar userId do runtime quando disponível", async () => {
    const state: GraphState = {
      messages: [],
      extractedPreferences: { name: "João", favoriteGenres: ["rock"] },
    };
    const runtime = { context: { userId: "user-123" } } as unknown as Runtime;

    const result = await savePreferencesNode(state, runtime);

    expect(mockMergePreferences).toHaveBeenCalledWith("user-123", {
      name: "João",
      favoriteGenres: ["rock"],
    });
    expect(result.extractedPreferences).toBeUndefined();
  });

  it("deve usar userId do state quando runtime não tem userId", async () => {
    const state: GraphState = {
      messages: [],
      extractedPreferences: { name: "Maria", favoriteGenres: ["pop"] },
      userId: "user-456",
    };
    const runtime = { context: {} } as unknown as Runtime;

    const result = await savePreferencesNode(state, runtime);

    expect(mockMergePreferences).toHaveBeenCalledWith("user-456", {
      name: "Maria",
      favoriteGenres: ["pop"],
    });
    expect(result.extractedPreferences).toBeUndefined();
  });

  it("deve usar userId do state quando runtime é omitido", async () => {
    const state: GraphState = {
      messages: [],
      extractedPreferences: { name: "Ana" },
      userId: "user-789",
    };

    const result = await savePreferencesNode(state);

    expect(mockMergePreferences).toHaveBeenCalledWith("user-789", {
      name: "Ana",
    });
    expect(result.extractedPreferences).toBeUndefined();
  });

  it('deve usar fallback "unknowm" quando nenhum userId é fornecido', async () => {
    const state: GraphState = {
      messages: [],
      extractedPreferences: { name: "Carlos" },
    };

    const result = await savePreferencesNode(state);

    expect(mockMergePreferences).toHaveBeenCalledWith("unknowm", {
      name: "Carlos",
    });
    expect(result.extractedPreferences).toBeUndefined();
  });

  it("deve propagar erro quando mergePreferences falha", async () => {
    mockMergePreferences.mockRejectedValue(new Error("DB error"));

    const state: GraphState = {
      messages: [],
      extractedPreferences: { name: "João" },
      userId: "user-1",
    };

    await expect(savePreferencesNode(state)).rejects.toThrow("DB error");
  });

  it("deve passar dados completos de extractedPreferences para mergePreferences", async () => {
    const prefs = {
      name: "Pedro",
      age: 25,
      favoriteGenres: ["rock", "mpb"],
      favoriteBands: ["Legião Urbana"],
      mood: "feliz",
      listeningContext: "em casa",
      additionalInfo: "tocando violão",
    };

    const state: GraphState = {
      messages: [],
      extractedPreferences: prefs,
      userId: "user-1",
    };

    await savePreferencesNode(state);

    expect(mockMergePreferences).toHaveBeenCalledWith("user-1", prefs);
  });
});
