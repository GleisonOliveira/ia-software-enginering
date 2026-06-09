import { jest, describe, it, expect, beforeEach, afterEach } from "@jest/globals";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store";
import { MemoryService } from "../../src/services/memoryService.ts";
import type { ModelConfig } from "../../src/config.ts";

const TEST_DB_URI = "postgresql://test:test@localhost:5432/test_db";

function createMockCheckpointer(): jest.Mocked<PostgresSaver> {
  return { setup: jest.fn() } as unknown as jest.Mocked<PostgresSaver>;
}

function createMockStore(): jest.Mocked<PostgresStore> {
  return { setup: jest.fn() } as unknown as jest.Mocked<PostgresStore>;
}

describe("MemoryService", () => {
  let mockCheckpointer: jest.Mocked<PostgresSaver>;
  let mockStore: jest.Mocked<PostgresStore>;

  beforeEach(() => {
    mockCheckpointer = createMockCheckpointer();
    mockStore = createMockStore();

    jest
      .spyOn(PostgresSaver, "fromConnString")
      .mockReturnValue(mockCheckpointer);
    jest
      .spyOn(PostgresStore, "fromConnString")
      .mockReturnValue(mockStore);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe("construtor", () => {
    it("deve usar a config padrão quando nenhum override é fornecido", () => {
      new MemoryService();

      expect(PostgresStore.fromConnString).toHaveBeenCalledTimes(1);
      expect(PostgresSaver.fromConnString).toHaveBeenCalledTimes(1);
    });

    it("deve usar configOverrides quando fornecido", () => {
      const overrides: ModelConfig = {
        apiKey: "custom-key",
        baseURL: "https://custom.api.com",
        httpReferer: "",
        xTitle: "Custom",
        models: ["custom-model"],
        temperature: 0.5,
        provider: { sort: { by: "throughput", partition: "none" } },
        memory: { dbUri: TEST_DB_URI },
      };

      new MemoryService(overrides);

      expect(PostgresStore.fromConnString).toHaveBeenCalledWith(TEST_DB_URI);
      expect(PostgresSaver.fromConnString).toHaveBeenCalledWith(TEST_DB_URI);
    });
  });

  describe("checkpointer getter", () => {
    it("deve retornar a instância do checkpointer", () => {
      const service = new MemoryService();
      expect(service.checkpointer).toBe(mockCheckpointer);
    });
  });

  describe("store getter", () => {
    it("deve retornar a instância da store", () => {
      const service = new MemoryService();
      expect(service.store).toBe(mockStore);
    });
  });

  describe("setup", () => {
    it("deve chamar setup da store e do checkpointer", async () => {
      const service = new MemoryService();
      await service.setup();

      expect(mockStore.setup).toHaveBeenCalledTimes(1);
      expect(mockCheckpointer.setup).toHaveBeenCalledTimes(1);
    });

    it("deve lançar erro se store.setup falhar", async () => {
      mockStore.setup.mockRejectedValue(new Error("store setup failed"));
      const service = new MemoryService();

      await expect(service.setup()).rejects.toThrow("store setup failed");

      expect(mockStore.setup).toHaveBeenCalledTimes(1);
      expect(mockCheckpointer.setup).not.toHaveBeenCalled();
    });

    it("deve lançar erro se checkpointer.setup falhar", async () => {
      mockCheckpointer.setup.mockRejectedValue(new Error("checkpointer setup failed"));
      const service = new MemoryService();

      await expect(service.setup()).rejects.toThrow("checkpointer setup failed");
    });
  });
});
