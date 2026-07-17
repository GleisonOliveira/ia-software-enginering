/**
 * Unit tests for tool-builder.ts — tool construction from skill contracts.
 *
 * Domain: tools
 *
 * Tests ToolBuilder class:
 * - Building a tool from a skill definition (mock fallback)
 * - Building a tool with LLM client
 * - Tool entry creation with metadata
 */

import { describe, it, expect } from "@jest/globals";
import { ToolBuilder } from "../../src/tools/tool-builder.js";
import { Logger } from "../../src/shared/logger.js";
import type { Skill } from "../../src/contracts/contracts.types.js";

/**
 * Creates a minimal skill definition for testing.
 */
function createTestSkill(overrides?: Partial<Skill>): Skill {
  return {
    nome: "test_tool",
    descricao: "A test tool that returns test data",
    entrada: { query: "string" },
    saida: { result: "string" },
    ...overrides,
  };
}

describe("ToolBuilder", () => {
  const logger = new Logger("error");
  const builder = new ToolBuilder(logger);

  describe("build", () => {
    it("returns an executable function from a skill definition", () => {
      const skill = createTestSkill();
      const toolFn = builder.build(skill);
      expect(typeof toolFn).toBe("function");
    });

    it("tool function returns mock data without LLM client", async () => {
      const skill = createTestSkill();
      const toolFn = builder.build(skill);
      const result = await toolFn({ query: "test" });

      expect(result.sucesso).toBe(true);
      expect(result.dados).toBeDefined();
      expect(result.dados["result"]).toBe("result_fallback");
      expect(result._tokens.total).toBe(0);
    });

    it("tool function handles list output types in fallback", async () => {
      const skill = createTestSkill({
        saida: { items: "list" },
      });
      const toolFn = builder.build(skill);
      const result = await toolFn({});

      expect(result.sucesso).toBe(true);
      expect(Array.isArray(result.dados["items"])).toBe(true);
    });

    it("tool function handles object output types in fallback", async () => {
      const skill = createTestSkill({
        saida: { metadata: "object" },
      });
      const toolFn = builder.build(skill);
      const result = await toolFn({});

      expect(result.sucesso).toBe(true);
      expect(typeof result.dados["metadata"]).toBe("object");
    });

    it("tool function handles int output types in fallback", async () => {
      const skill = createTestSkill({
        saida: { count: "int" },
      });
      const toolFn = builder.build(skill);
      const result = await toolFn({});

      expect(result.sucesso).toBe(true);
      expect(result.dados["count"]).toBe(42);
    });
  });

  describe("buildEntry", () => {
    it("returns a complete ToolRegistryEntry", () => {
      const skill = createTestSkill();
      const entry = builder.buildEntry(skill);

      expect(entry.skill).toBe(skill);
      expect(entry.definition.name).toBe("test_tool");
      expect(entry.definition.description).toBe("A test tool that returns test data");
      expect(entry.definition.inputSchema).toEqual({ query: "string" });
      expect(entry.definition.outputSchema).toEqual({ result: "string" });
      expect(typeof entry.definition.fn).toBe("function");
    });

    it("entry fn is executable", async () => {
      const skill = createTestSkill();
      const entry = builder.buildEntry(skill);
      const result = await entry.definition.fn({ query: "test" });
      expect(result.sucesso).toBe(true);
    });
  });
});
