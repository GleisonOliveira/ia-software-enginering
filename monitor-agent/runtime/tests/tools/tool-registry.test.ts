/**
 * Unit tests for tool-registry.ts — tool lookup and registration.
 *
 * Domain: tools
 *
 * Tests the ToolRegistry class:
 * - Registration and lookup
 * - Multiple registrations
 * - getNames, getSkills, size, clear
 * - toToolMap conversion
 */

import { describe, it, expect, beforeEach } from "@jest/globals";
import { ToolRegistry } from "../../src/tools/tool-registry.js";
import type { ToolRegistryEntry } from "../../src/tools/tools.types.js";
import type { ActionResult } from "../../src/shared/shared.types.js";

/**
 * Creates a mock ToolRegistryEntry for testing.
 */
function createMockEntry(name: string): ToolRegistryEntry {
  const mockFn = async (args: Record<string, unknown>): Promise<ActionResult> => ({
    sucesso: true,
    dados: { result: `${name}_output` },
    erro: "",
    _tokens: { prompt: 0, completion: 0, total: 0 },
    _entrada: args,
  });

  return {
    skill: {
      nome: name,
      descricao: `Tool ${name}`,
      entrada: { input: "string" },
      saida: { output: "string" },
    },
    definition: {
      name,
      description: `Tool ${name}`,
      inputSchema: { input: "string" },
      outputSchema: { output: "string" },
      fn: mockFn,
    },
  };
}

describe("ToolRegistry", () => {
  let registry: ToolRegistry;

  beforeEach(() => {
    registry = new ToolRegistry();
  });

  describe("register and get", () => {
    it("registers and retrieves a tool", () => {
      const entry = createMockEntry("search");
      registry.register(entry);
      expect(registry.get("search")).toBe(entry);
    });

    it("returns undefined for unknown tool", () => {
      expect(registry.get("unknown")).toBeUndefined();
    });

    it("overwrites existing entry with same name", () => {
      const entry1 = createMockEntry("search");
      const entry2 = createMockEntry("search");
      registry.register(entry1);
      registry.register(entry2);
      expect(registry.get("search")).toBe(entry2);
    });
  });

  describe("has", () => {
    it("returns true for registered tool", () => {
      registry.register(createMockEntry("search"));
      expect(registry.has("search")).toBe(true);
    });

    it("returns false for unknown tool", () => {
      expect(registry.has("search")).toBe(false);
    });
  });

  describe("registerAll", () => {
    it("registers multiple tools at once", () => {
      const entries = [createMockEntry("a"), createMockEntry("b"), createMockEntry("c")];
      registry.registerAll(entries);
      expect(registry.size()).toBe(3);
      expect(registry.has("a")).toBe(true);
      expect(registry.has("b")).toBe(true);
      expect(registry.has("c")).toBe(true);
    });
  });

  describe("getNames", () => {
    it("returns all registered tool names", () => {
      registry.register(createMockEntry("x"));
      registry.register(createMockEntry("y"));
      const names = registry.getNames();
      expect(names).toContain("x");
      expect(names).toContain("y");
      expect(names).toHaveLength(2);
    });

    it("returns empty array for empty registry", () => {
      expect(registry.getNames()).toHaveLength(0);
    });
  });

  describe("getSkills", () => {
    it("returns skill objects from registered entries", () => {
      registry.register(createMockEntry("search"));
      const skills = registry.getSkills();
      expect(skills).toHaveLength(1);
      expect(skills[0]?.nome).toBe("search");
    });
  });

  describe("getFunction", () => {
    it("returns the executable function for a tool", () => {
      const entry = createMockEntry("search");
      registry.register(entry);
      const fn = registry.getFunction("search");
      expect(fn).toBeDefined();
      expect(fn).toBe(entry.definition.fn);
    });

    it("returns undefined for unknown tool", () => {
      expect(registry.getFunction("unknown")).toBeUndefined();
    });
  });

  describe("toToolMap", () => {
    it("converts registry to Map<string, ToolFunction>", () => {
      const entry1 = createMockEntry("a");
      const entry2 = createMockEntry("b");
      registry.register(entry1);
      registry.register(entry2);

      const map = registry.toToolMap();
      expect(map.size).toBe(2);
      expect(map.get("a")).toBe(entry1.definition.fn);
      expect(map.get("b")).toBe(entry2.definition.fn);
    });
  });

  describe("clear", () => {
    it("removes all registered tools", () => {
      registry.register(createMockEntry("a"));
      registry.register(createMockEntry("b"));
      registry.clear();
      expect(registry.size()).toBe(0);
      expect(registry.get("a")).toBeUndefined();
    });
  });

  describe("size", () => {
    it("returns correct count", () => {
      expect(registry.size()).toBe(0);
      registry.register(createMockEntry("a"));
      expect(registry.size()).toBe(1);
      registry.register(createMockEntry("b"));
      expect(registry.size()).toBe(2);
    });
  });
});
