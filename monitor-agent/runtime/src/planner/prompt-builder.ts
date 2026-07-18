/**
 * Prompt builder — constructs the system prompt from contracts.
 *
 * Domain: planner
 *
 * Builds the LLM system prompt by combining agent metadata, tool descriptions,
 * planner rules, policies, and mode-specific instructions from contracts.
 * The prompt instructs the LLM on its role, available tools, response format,
 * and constraints. Mirrors the Python runtime's construir_prompt_sistema().
 *
 * Used by: planner module, cycle runner.
 */

import type { AllContracts } from "../contracts/contracts.types.js";

/**
 * Mode-specific instruction blocks appended to the system prompt.
 *
 * Each agent type receives tailored instructions on how to behave during
 * the planning phase. The block is inserted into the prompt only when
 * the agent type matches.
 */
const MODE_INSTRUCTIONS: Record<string, string> = {
  interactive: `
MODO INTERACTIVE:
- Antes de agir, valide ambiguidades com o usuario usando PERGUNTAR_USUARIO
- Se faltar informacao critica, pergunte antes de chamar ferramentas
- Inclua o campo "pergunta" com a pergunta para o usuario
`,
  goal_oriented: `
MODO GOAL-ORIENTED:
- Decomponha o objetivo em sub-objetivos executaveis
- Para cada sub-objetivo, planeje quais ferramentas usar
- Reavalie o plano apos cada etapa com base nos resultados
`,
  autonomous: `
MODO AUTONOMOUS:
- Responda ao evento trigger fornecido na percepcao
- Opere dentro dos limites rigidos definidos
- NUNCA execute acoes destrutivas sem confirmacao humana
- Priorize seguranca sobre velocidade
`,
};

/**
 * Builds the LLM system prompt from contract definitions.
 *
 * Assembles the system prompt by combining:
 * - Agent identity (name, description, type, objective)
 * - Available tools with input/output schemas
 * - Response format specification (JSON schema)
 * - Planner rules from the planner contract
 * - Agent policies from the rules contract
 * - Mode-specific instructions
 *
 * Used by: Planner.plan(), Planner.mockPlan().
 */
export class PromptBuilder {
  /**
   * Builds the complete system prompt from contracts.
   *
   * @param contracts - The loaded and validated contract set.
   * @returns The complete system prompt string.
   *
   * Used by: planner module for LLM calls.
   *
   * Acceptance criteria:
   * - Includes instructions for all modes (interactive, goal_oriented, autonomous).
   */
  build(contracts: AllContracts): string {
    const agente = contracts.agente;
    const ciclo = contracts.ciclo;
    const habilidades = contracts.habilidades.habilidades;
    const planejador = contracts.planejador;
    const regras = contracts.regras;

    // Agent identity
    const nomeAgente = agente.nome;
    const descricaoAgente = agente.descricao;
    const tipoAgente = agente.tipo;
    const objetivo = ciclo.objetivo;

    // Build tools block
    const blocoFerramentas = PromptBuilder.buildToolsBlock(habilidades);

    // Planner rules
    const textoRegras = PromptBuilder.buildListBlock(planejador.regras);

    // Agent policies
    const textoPoliticas = PromptBuilder.buildListBlock(regras.politicas);

    // Mode-specific instructions
    const instrucoesTipo = MODE_INSTRUCTIONS[tipoAgente] ?? "";

    return `Voce e o planejador de um agente autonomo.

Agente: ${nomeAgente} - ${descricaoAgente}
Tipo: ${tipoAgente}
Objetivo: ${objetivo}

Etapas do ciclo: perceber -> planejar -> agir -> avaliar

Ferramentas disponiveis:
${blocoFerramentas}
Formato de resposta (APENAS JSON valido):
{
  "proxima_acao": "CHAMAR_FERRAMENTA" ou "FINALIZAR" ou "PERGUNTAR_USUARIO",
  "nome_ferramenta": "nome da ferramenta (obrigatorio se CHAMAR_FERRAMENTA)",
  "argumentos_ferramenta": {},
  "criterio_sucesso": "o que define sucesso para esta etapa",
  "pergunta": "pergunta para o usuario (obrigatorio se PERGUNTAR_USUARIO)"
}

CRITICO: o campo "proxima_acao" DEVE ser exatamente um destes 3 valores:
- "CHAMAR_FERRAMENTA" — para executar uma ferramenta
- "FINALIZAR" — para encerrar o ciclo
- "PERGUNTAR_USUARIO" — para pedir informacao ao usuario
NUNCA use o nome da ferramenta como proxima_acao. Use "CHAMAR_FERRAMENTA" e coloque o nome em "nome_ferramenta".

Regras gerais:
- Use cada ferramenta no maximo uma vez, a menos que precise de parametros diferentes
- As chaves de argumentos_ferramenta devem corresponder exatamente aos campos de entrada da ferramenta
- Para campos do tipo object, use dados reais coletados nas etapas anteriores
${instrucoesTipo}
IMPORTANTE — Regras do planejador (voce DEVE seguir TODAS):
${textoRegras}

IMPORTANTE — Politicas do agente (voce DEVE seguir TODAS):
${textoPoliticas}

ATENCAO: voce NAO pode usar FINALIZAR enquanto alguma regra ou politica acima nao for satisfeita.
Se uma regra exige chamar uma ferramenta antes de finalizar, voce DEVE chama-la primeiro.
`;
  }

  /**
   * Builds the tools description block from skill contracts.
   *
   * Each skill is formatted as:
   * - nome: descricao
   *   entrada: {campo: tipo, ...}
   *   saida: {campo: tipo, ...}
   *
   * @param habilidades - Array of skill definitions from the contracts.
   * @returns Formatted tools block string.
   *
   * Used by: build().
   */
  private static buildToolsBlock(
    habilidades: readonly { readonly nome: string; readonly descricao: string; readonly entrada: Record<string, string>; readonly saida: Record<string, string> }[],
  ): string {
    if (habilidades.length === 0) {
      return "- nenhuma ferramenta disponivel\n";
    }

    return habilidades
      .map((habilidade) => {
        const textoEntradas = PromptBuilder.formatSchemaFields(habilidade.entrada);
        const textoSaidas = PromptBuilder.formatSchemaFields(habilidade.saida);
        return `- ${habilidade.nome}: ${habilidade.descricao}\n  entrada: {${textoEntradas}}\n  saida: {${textoSaidas}}`;
      })
      .join("\n") + "\n";
  }

  /**
   * Formats a schema record (param -> type) as "campo: tipo, ..." string.
   *
   * @param schema - The parameter schema record.
   * @returns Formatted string, or "nenhuma" if empty.
   *
   * Used by: buildToolsBlock().
   */
  private static formatSchemaFields(schema: Record<string, string>): string {
    const entries = Object.entries(schema);
    if (entries.length === 0) return "nenhuma";
    return entries.map(([nome, tipo]) => `${nome}: ${tipo}`).join(", ");
  }

  /**
   * Builds a numbered list block from an array of strings.
   *
   * @param items - Array of rule/policy strings.
   * @returns Formatted list block, or empty string if empty.
   *
   * Used by: build() for planner rules and agent policies.
   */
  private static buildListBlock(items: readonly string[]): string {
    if (items.length === 0) return "";
    return items.map((item) => `- ${item}`).join("\n");
  }
}
