/**
 * Unit tests for cli/commands/validate.ts — ValidateCommand.
 *
 * Domain: cli
 *
 * Tests ValidateCommand.execute() with valid and invalid contracts.
 */

import { describe, it, expect, jest, beforeEach } from "@jest/globals";
import { ValidateCommand } from "../../../src/cli/commands/validate.js";
import { Logger } from "../../../src/shared/logger.js";
import type { AllContracts } from "../../../src/contracts/contracts.types.js";

function createValidContracts(): AllContracts {
  return {
    agente: { nome: "Test", descricao: "Test", tipo: "task_based", objetivo: "Test", contrato_saida: { formato: "json", campos_obrigatorios: [], exemplo: {} } },
    ciclo: { objetivo: "Test", ciclo: { max_etapas: 10 }, condicoes_parada: [] },
    planejador: { formato_saida: { proxima_acao: "CHAMAR_FERRAMENTA", criterio_sucesso: "string" }, regras: [] },
    caixa_ferramentas: { ferramentas: [] },
    executor: { execucao: { validar_entrada: false, tentar_novamente_em_falha: false }, pos_execucao: { avaliar_resultado: false } },
    regras: { ferramentas_obrigatorias: [], limites: { max_etapas: 10, sem_progresso: 3, limite_tempo_segundos: 300, chamadas_ferramenta: {} }, acoes_sensiveis: [], politicas: [] },
    ganchos: { ganchos: { antes_da_etapa: "log", apos_etapa: "log", antes_da_acao: "log", apos_acao: "log", em_erro: "alerta" } },
    habilidades: { habilidades: [] },
    memoria: { memoria_curta: { guardar: [], descartar: [], max_registros: 10 }, resumo_final: { max_linhas: 5, campos: [] } },
  };
}

describe("ValidateCommand", () => {
  const logger = new Logger("error");

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("reports success for valid contracts", () => {
    const mockLoader = {
      loadAllContracts: jest.fn<() => AllContracts>().mockReturnValue(createValidContracts()),
    };
    const command = new ValidateCommand(mockLoader as never, logger);

    // Should not throw
    expect(() => command.execute({ agente: "/path" })).not.toThrow();
    expect(mockLoader.loadAllContracts).toHaveBeenCalledWith("/path");
  });

  it("exits with code 1 for invalid contracts", () => {
    const mockLoader = {
      loadAllContracts: jest.fn<() => AllContracts>().mockImplementation(() => {
        throw new Error("Invalid contracts");
      }),
    };
    const command = new ValidateCommand(mockLoader as never, logger);

    const mockExit = jest.spyOn(process, "exit").mockImplementation(() => undefined as never);
    command.execute({ agente: "/path" });
    expect(mockExit).toHaveBeenCalledWith(1);
    mockExit.mockRestore();
  });
});
