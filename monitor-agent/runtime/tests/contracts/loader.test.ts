/**
 * Unit tests for loader.ts — YAML extraction from .md contract files.
 *
 * Domain: contracts
 *
 * Tests the loadYamlFromMd() function with various .md file scenarios
 * and loadAllContracts() with a complete agent directory.
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { loadYamlFromMd, loadAllContracts } from "../../src/contracts/loader.js";

describe("loader", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "agent-test-"));
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  describe("loadYamlFromMd", () => {
    it("extracts YAML from a .md file with a fenced code block", () => {
      const md = `# Contract

Some description text.

\`\`\`yaml
nome: test-agent
descricao: A test agent
tipo: task_based
objetivo: Test objective
contrato_saida:
  formato: json
  campos_obrigatorios:
    - result
  exemplo:
    result: "ok"
\`\`\`\n`;
      const filePath = path.join(tmpDir, "test.md");
      fs.writeFileSync(filePath, md);

      const result = loadYamlFromMd(filePath);
      expect(result).toEqual({
        nome: "test-agent",
        descricao: "A test agent",
        tipo: "task_based",
        objetivo: "Test objective",
        contrato_saida: {
          formato: "json",
          campos_obrigatorios: ["result"],
          exemplo: { result: "ok" },
        },
      });
    });

    it("returns empty object when file does not exist", () => {
      const result = loadYamlFromMd("/nonexistent/path/file.md");
      expect(result).toEqual({});
    });

    it("returns empty object when file has no YAML block", () => {
      const filePath = path.join(tmpDir, "noyaml.md");
      fs.writeFileSync(filePath, "# Just a heading\n\nNo code block here.\n");

      const result = loadYamlFromMd(filePath);
      expect(result).toEqual({});
    });

    it("returns the first YAML block when multiple exist", () => {
      const md = `\`\`\`yaml
first: true
\`\`\`\n\nSome text.\n\n\`\`\`yaml\nsecond: true\n\`\`\`\n`;
      const filePath = path.join(tmpDir, "multi.md");
      fs.writeFileSync(filePath, md);

      const result = loadYamlFromMd(filePath);
      expect(result).toEqual({ first: true });
    });

    it("handles YAML with nested structures", () => {
      const md = `\`\`\`yaml
limites:
  max_etapas: 10
  sem_progresso: 3
  limite_tempo_segundos: 120
  chamadas_ferramenta:
    search: 5
    total: "ilimitado"
\`\`\`\n`;
      const filePath = path.join(tmpDir, "nested.md");
      fs.writeFileSync(filePath, md);

      const result = loadYamlFromMd(filePath);
      expect(result).toEqual({
        limites: {
          max_etapas: 10,
          sem_progresso: 3,
          limite_tempo_segundos: 120,
          chamadas_ferramenta: {
            search: 5,
            total: "ilimitado",
          },
        },
      });
    });

    it("returns empty object for invalid YAML", () => {
      const md = `\`\`\`yaml
invalid: [unclosed
\`\`\`\n`;
      const filePath = path.join(tmpDir, "bad.yaml");
      fs.writeFileSync(filePath, md);

      const result = loadYamlFromMd(filePath);
      expect(result).toEqual({});
    });
  });

  describe("loadAllContracts", () => {
    /**
     * Creates a complete agent directory with all 9 contract .md files.
     * Uses minimal valid contracts for each schema.
     */
    function createAgentDir(): string {
      const agentDir = path.join(tmpDir, "agent");
      const contractsDir = path.join(agentDir, "contracts");
      fs.mkdirSync(contractsDir, { recursive: true });

      fs.writeFileSync(
        path.join(agentDir, "agent.md"),
        `\`\`\`yaml
nome: test-agent
descricao: A test agent for unit tests
tipo: task_based
objetivo: Test the runtime
contrato_saida:
  formato: json
  campos_obrigatorios:
    - result
  exemplo:
    result: "ok"
\`\`\`\n`,
      );

      fs.writeFileSync(
        path.join(contractsDir, "loop.md"),
        `\`\`\`yaml
objetivo: Test the runtime
ciclo:
  max_etapas: 5
condicoes_parada:
  - objective achieved
\`\`\`\n`,
      );

      fs.writeFileSync(
        path.join(contractsDir, "planner.md"),
        `\`\`\`yaml
formato_saida:
  proxima_acao: CHAMAR_FERRAMENTA
  criterio_sucesso: Tool executed
regras:
  - Always use tools
\`\`\`\n`,
      );

      fs.writeFileSync(
        path.join(contractsDir, "toolbox.md"),
        `\`\`\`yaml
ferramentas:
  - nome: search
    entrada:
      query: string
\`\`\`\n`,
      );

      fs.writeFileSync(
        path.join(contractsDir, "executor.md"),
        `\`\`\`yaml
execucao:
  validar_entrada: true
  tentar_novamente_em_falha: false
pos_execucao:
  avaliar_resultado: true
\`\`\`\n`,
      );

      fs.writeFileSync(
        path.join(agentDir, "rules.md"),
        `\`\`\`yaml
ferramentas_obrigatorias: []
limites:
  max_etapas: 10
  sem_progresso: 3
  limite_tempo_segundos: 120
  chamadas_ferramenta:
    total: "ilimitado"
acoes_sensiveis: []
politicas: []
\`\`\`\n`,
      );

      fs.writeFileSync(
        path.join(agentDir, "hooks.md"),
        `\`\`\`yaml
ganchos:
  antes_da_etapa: log
  apos_etapa: log
  antes_da_acao: log
  apos_acao: log
  em_erro: alerta
\`\`\`\n`,
      );

      fs.writeFileSync(
        path.join(agentDir, "skills.md"),
        `\`\`\`yaml
habilidades:
  - nome: search
    descricao: Search for information
    entrada:
      query: string
    saida:
      results: string
\`\`\`\n`,
      );

      fs.writeFileSync(
        path.join(agentDir, "memory.md"),
        `\`\`\`yaml
memoria_curta:
  guardar:
    - context
  descartar:
    - temp
  max_registros: 10
resumo_final:
  max_linhas: 20
  campos:
    - result
\`\`\`\n`,
      );

      return agentDir;
    }

    it("loads all 9 contracts from a valid agent directory", () => {
      const agentDir = createAgentDir();
      const contracts = loadAllContracts(agentDir);

      expect(contracts.agente.nome).toBe("test-agent");
      expect(contracts.agente.tipo).toBe("task_based");
      expect(contracts.ciclo.ciclo.max_etapas).toBe(5);
      expect(contracts.planejador.regras).toEqual(["Always use tools"]);
      expect(contracts.caixa_ferramentas.ferramentas).toHaveLength(1);
      expect(contracts.executor.execucao.validar_entrada).toBe(true);
      expect(contracts.regras.limites.max_etapas).toBe(10);
      expect(contracts.ganchos.ganchos.antes_da_etapa).toBe("log");
      expect(contracts.habilidades.habilidades).toHaveLength(1);
      expect(contracts.memoria.memoria_curta.max_registros).toBe(10);
    });

    it("throws when a required contract file is missing", () => {
      const agentDir = createAgentDir();
      // Remove the agent contract
      fs.unlinkSync(path.join(agentDir, "agent.md"));

      expect(() => loadAllContracts(agentDir)).toThrow("Invalid contracts");
    });
  });
});
