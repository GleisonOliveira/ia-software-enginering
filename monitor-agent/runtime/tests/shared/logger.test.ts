/**
 * Unit tests for logger.ts — structured logger with phase-aware formatting.
 *
 * Domain: shared
 *
 * Tests the Logger class including log level filtering, phase prefixes,
 * child logger creation, and output formatting.
 */

import { jest } from "@jest/globals";
import { Logger } from "../../src/shared/logger.js";

describe("Logger", () => {
  let stdoutWriteSpy: jest.SpyInstance;
  let stderrWriteSpy: jest.SpyInstance;

  beforeEach(() => {
    stdoutWriteSpy = jest.spyOn(process.stdout, "write").mockImplementation(() => true);
    stderrWriteSpy = jest.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    stdoutWriteSpy.mockRestore();
    stderrWriteSpy.mockRestore();
  });

  describe("log level filtering", () => {
    it("outputs all messages when level is debug", () => {
      const log = new Logger("debug");
      log.debug("d1");
      log.info("i1");
      log.warn("w1");
      log.error("e1");

      // debug goes to stdout, info goes to stdout, warn goes to stderr, error goes to stderr
      expect(stdoutWriteSpy).toHaveBeenCalledTimes(2);
      expect(stderrWriteSpy).toHaveBeenCalledTimes(2);
    });

    it("filters debug messages when level is info", () => {
      const log = new Logger("info");
      log.debug("should-not-appear");
      log.info("visible");

      expect(stdoutWriteSpy).toHaveBeenCalledTimes(1);
      const output = stdoutWriteSpy.mock.calls[0][0] as string;
      expect(output).toContain("visible");
    });

    it("filters debug and info when level is warn", () => {
      const log = new Logger("warn");
      log.debug("no");
      log.info("no");
      log.warn("yes");
      log.error("yes");

      expect(stdoutWriteSpy).toHaveBeenCalledTimes(0);
      expect(stderrWriteSpy).toHaveBeenCalledTimes(2);
    });

    it("only outputs errors when level is error", () => {
      const log = new Logger("error");
      log.debug("no");
      log.info("no");
      log.warn("no");
      log.error("yes");

      expect(stdoutWriteSpy).toHaveBeenCalledTimes(0);
      expect(stderrWriteSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe("phase prefix", () => {
    it("prepends phase name in brackets", () => {
      const log = new Logger("debug", "perceber");
      log.info("step started");

      const output = stdoutWriteSpy.mock.calls[0][0] as string;
      expect(output).toBe("[perceber] step started\n");
    });

    it("omits prefix when no phase is set", () => {
      const log = new Logger("debug");
      log.info("general message");

      const output = stdoutWriteSpy.mock.calls[0][0] as string;
      expect(output).toBe("general message\n");
    });
  });

  describe("child logger", () => {
    it("creates a child with phase prefix", () => {
      const parent = new Logger("debug");
      const child = parent.child("planejar");

      child.info("planning");

      const output = stdoutWriteSpy.mock.calls[0][0] as string;
      expect(output).toBe("[planejar] planning\n");
    });

    it("child inherits parent log level", () => {
      const parent = new Logger("warn");
      const child = parent.child("agir");

      child.info("should not appear");
      child.warn("visible");

      expect(stdoutWriteSpy).toHaveBeenCalledTimes(0);
      expect(stderrWriteSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe("structured data", () => {
    it("appends JSON data after pipe separator", () => {
      const log = new Logger("debug");
      log.info("tool called", { tool: "search", args: { q: "test" } });

      const output = stdoutWriteSpy.mock.calls[0][0] as string;
      expect(output).toContain(' | {"tool":"search","args":{"q":"test"}}');
    });

    it("omits pipe when no data is provided", () => {
      const log = new Logger("debug");
      log.info("simple message");

      const output = stdoutWriteSpy.mock.calls[0][0] as string;
      expect(output).toBe("simple message\n");
    });
  });

  describe("setLevel", () => {
    it("changes the log level dynamically", () => {
      const log = new Logger("error");
      log.info("hidden");
      expect(stdoutWriteSpy).toHaveBeenCalledTimes(0);

      log.setLevel("info");
      log.info("visible");
      expect(stdoutWriteSpy).toHaveBeenCalledTimes(1);
    });
  });
});
