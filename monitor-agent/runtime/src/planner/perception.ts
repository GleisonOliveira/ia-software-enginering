/**
 * Perception builder — constructs the context prompt from agent state.
 *
 * Domain: planner
 *
 * Builds a multi-line string summarizing the current agent state for the LLM
 * planner. Includes the user input (alert), agent mode, event context,
 * history of previous steps, tools used, step/token counts, and stagnation
 * warnings. Mirrors the Python runtime's perceber() function.
 *
 * Used by: cycle runner (perceive phase), planner module.
 */

import type { AgentState } from "../core/state.types.js";

/**
 * Builds perception prompts from agent state.
 *
 * The perception string is fed to the LLM as the user prompt during planning.
 * It provides a concise summary of everything the LLM needs to decide the
 * next action: what happened so far, what tools were called, and what
 * resource limits remain.
 *
 * Used by: CycleRunner (perceive phase).
 */
export class PerceptionBuilder {
  /**
   * Builds the perception prompt from the current agent state.
   *
   * Constructs a multi-line string with:
   * - The user input (alerta)
   * - Agent mode (tipo_agente)
   * - Event trigger (if present)
   * - History entries with tool results
   * - Tools already used
   * - Step and tool call counts
   * - Stagnation warnings
   *
   * @param state - The current agent state (immutable snapshot).
   * @returns The perception prompt string.
   *
   * Used by: cycle runner (perceive phase).
   *
   * Acceptance criteria:
   * - Returns a string with entry, mode, and progress info.
   */
  build(state: AgentState): string {
    const parts: string[] = [];

    // User input (alerta)
    parts.push(`Alerta: ${state.entrada}`);

    // Agent mode
    parts.push(`Modo: ${state.tipoAgente}`);

    // Event trigger (if present)
    if (state.evento) {
      parts.push(`Evento trigger: ${state.evento}`);
    }

    // History entries with tool results
    for (const registro of state.historico) {
      const etapa = registro.etapa;
      const plano = registro.plano;
      const ferramentaUsada = plano.nomeFerramenta ?? "nenhuma";

      if (registro.resultadoAcao) {
        const resultado = PerceptionBuilder.formatActionResult(registro.resultadoAcao);
        parts.push(`Etapa ${etapa} [${ferramentaUsada}]: ${resultado}`);
      }
    }

    // Tools already used
    const ferramentasUsadas = Object.keys(state.chamadasPorFerramenta);
    if (ferramentasUsadas.length > 0) {
      parts.push(`Ferramentas ja utilizadas: ${ferramentasUsadas.join(", ")}`);
    }

    // Step and tool call counts
    parts.push(`Etapas realizadas: ${state.etapa}/${state.limits.maxEtapas}`);
    parts.push(`Chamadas de ferramenta: ${state.chamadasFerramenta}/${state.limits.maxChamadasFerramenta}`);

    // Stagnation warning
    if (state.etapasSemProgresso > 0) {
      parts.push(`ATENCAO: ${state.etapasSemProgresso} etapas sem progresso detectadas`);
    }

    return parts.join("\n");
  }

  /**
   * Formats an ActionResult for inclusion in the perception string.
   *
   * Extracts the success flag and relevant data fields, truncating
   * long output to keep the perception prompt concise.
   *
   * @param resultado - The action result from a previous step.
   * @returns A compact string representation of the result.
   *
   * Used by: build().
   */
  private static formatActionResult(resultado: {
    readonly sucesso: boolean;
    readonly dados: Record<string, unknown>;
    readonly erro: string;
  }): string {
    if (!resultado.sucesso) {
      return `FALHA: ${resultado.erro || "sem detalhes"}`;
    }

    // Format the data fields, excluding internal _-prefixed keys
    const relevantData: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(resultado.dados)) {
      if (!key.startsWith("_")) {
        relevantData[key] = value;
      }
    }

    const formatted = JSON.stringify(relevantData);
    // Truncate long results to keep perception concise
    return formatted.length > 200 ? formatted.substring(0, 200) + "..." : formatted;
  }
}
