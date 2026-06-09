import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store";
import { ModelConfig, config } from "../config.ts";

export class MemoryService {
  private _checkpointer: PostgresSaver;
  private _store: PostgresStore;
  private config: ModelConfig;

  constructor(configOverrides?: ModelConfig) {
    this.config = configOverrides ?? config;
    this._store = PostgresStore.fromConnString(this.config.memory.dbUri);
    this._checkpointer = PostgresSaver.fromConnString(this.config.memory.dbUri);
  }

  public get checkpointer() {
    return this._checkpointer;
  }

  public get store() {
    return this._store;
  }

  public async setup(): Promise<void> {
    await this._store.setup();
    await this._checkpointer.setup();
  }
}
