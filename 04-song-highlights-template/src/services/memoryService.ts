import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store";
import { ModelConfig, config } from "../config.ts";

export class MemoryService {
  private checkpointer: PostgresSaver;
  private store: PostgresStore;
  private config: ModelConfig;

  constructor(configOverrides?: ModelConfig) {
    this.config = configOverrides ?? config;
    this.store = PostgresStore.fromConnString(this.config.memory.dbUri);
  }
}
