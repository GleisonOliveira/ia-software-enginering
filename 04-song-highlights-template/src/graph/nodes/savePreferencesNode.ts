import type { Runtime } from "@langchain/langgraph";
import type { GraphState } from "../graph.ts";
import { type PreferencesService } from "../../services/preferencesService.ts";

export function createSavePreferencesNode(
  preferencesService: PreferencesService,
) {
  return async (
    state: GraphState,
    runtime?: Runtime,
  ): Promise<Partial<GraphState>> => {
    if (!state.extractedPreferences) {
      return {};
    }

    const userId = String(
      runtime?.context?.userId || state.userId || "unknowm",
    );

    await preferencesService.mergePreferences(
      userId,
      state.extractedPreferences,
    );

    return {
      extractedPreferences: undefined,
    };
  };
}
