import { OpenRouterService } from "../services/openrouterService.ts";
import { config } from "../config.ts";
import { buildChatGraph } from "./graph.ts";
import { MemoryService } from "../services/memoryService.ts";
import { PreferencesService } from "../services/preferencesService.ts";

export async function buildGraph(dbPath: string = "./preferences.db") {
  const llmClient = new OpenRouterService(config);
  const memoryService = new MemoryService(config);
  const preferenceService = new PreferencesService(dbPath);

  await memoryService.setup();

  const graph = buildChatGraph(llmClient, memoryService, preferenceService);

  return {
    graph,
    preferenceService,
    memoryService,
  };
}

export const graph = async () => buildGraph();
export default graph;
