/**
 * Tool registry for tool lookup by name.
 *
 * Domain: tools
 *
 * Maintains a Map of tool names to their ToolRegistryEntry, providing
 * efficient O(1) lookup by tool name during the act phase. Populated
 * from skill contracts via the tool builder.
 *
 * Used by: cycle runner (act phase), executor module, tool builder.
 */

import type { ToolRegistryEntry, ToolFunction } from "./tools.types.js";
import type { Skill } from "../contracts/contracts.types.js";

/**
 * Registry of executable tools keyed by name.
 *
 * Provides O(1) lookup for tool execution and metadata access.
 * Populated once at cycle start from skill contracts via buildToolEntry().
 *
 * Used by: cycle runner, executor module, tool existence checks.
 */
export class ToolRegistry {
  /** Internal map from tool name to its registry entry. */
  private readonly entries: Map<string, ToolRegistryEntry>;

  /**
   * Creates an empty ToolRegistry.
   */
  constructor() {
    this.entries = new Map();
  }

  /**
   * Registers a tool entry in the registry.
   *
   * Overwrites any existing entry with the same name (last-write-wins).
   * This allows re-registration if contracts are reloaded.
   *
   * @param entry - The tool registry entry to register.
   *
   * Used by: cycle runner initialization, tool builder.
   */
  register(entry: ToolRegistryEntry): void {
    this.entries.set(entry.definition.name, entry);
  }

  /**
   * Registers multiple tool entries at once.
   *
   * @param entries - Array of tool registry entries to register.
   *
   * Used by: cycle runner initialization with all skill contracts.
   */
  registerAll(entries: ToolRegistryEntry[]): void {
    for (const entry of entries) {
      this.register(entry);
    }
  }

  /**
   * Looks up a tool by name.
   *
   * @param name - The tool name to look up.
   * @returns The tool registry entry, or undefined if not found.
   *
   * Used by: executor module, circuit breaker, cycle runner.
   */
  get(name: string): ToolRegistryEntry | undefined {
    return this.entries.get(name);
  }

  /**
   * Checks if a tool exists in the registry.
   *
   * @param name - The tool name to check.
   * @returns True if the tool is registered.
   *
   * Used by: circuit breaker validation, cycle runner.
   */
  has(name: string): boolean {
    return this.entries.has(name);
  }

  /**
   * Returns the executable function for a tool by name.
   *
   * Convenience method that extracts the fn from the registry entry.
   *
   * @param name - The tool name.
   * @returns The ToolFunction, or undefined if not found.
   *
   * Used by: executor module for direct tool invocation.
   */
  getFunction(name: string): ToolFunction | undefined {
    return this.entries.get(name)?.definition.fn;
  }

  /**
   * Returns all registered tool names.
   *
   * @returns Array of registered tool names.
   *
   * Used by: circuit breaker for available tool listing, cycle runner.
   */
  getNames(): string[] {
    return [...this.entries.keys()];
  }

  /**
   * Returns all registered skills (contract definitions).
   *
   * @returns Array of Skill objects from registered entries.
   *
   * Used by: prompt builder for tool descriptions, validation.
   */
  getSkills(): Skill[] {
    return [...this.entries.values()].map((e) => e.skill);
  }

  /**
   * Returns the total number of registered tools.
   *
   * @returns Tool count.
   */
  size(): number {
    return this.entries.size;
  }

  /**
   * Clears all registered tools.
   *
   * Useful for testing or when reloading contracts mid-execution.
   */
  clear(): void {
    this.entries.clear();
  }

  /**
   * Converts the registry to a Map<string, ToolFunction> for executor use.
   *
   * The executor module expects a Map of tool functions rather than the
   * full registry. This method creates that map on demand.
   *
   * @returns Map from tool name to executable function.
   *
   * Used by: executor module, cycle runner tool execution.
   */
  toToolMap(): Map<string, ToolFunction> {
    const map = new Map<string, ToolFunction>();
    for (const [name, entry] of this.entries) {
      map.set(name, entry.definition.fn);
    }
    return map;
  }
}
