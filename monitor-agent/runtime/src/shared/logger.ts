/**
 * Structured logger replacing Python runtime print statements.
 *
 * Domain: shared
 *
 * Provides a minimal, phase-aware logger with consistent formatting.
 * Each log line is prefixed with the phase name in brackets (e.g., [perceber]),
 * matching the Python runtime's output convention for backward compatibility.
 *
 * Used by: Cycle runner phases, planner, executor, tool builder, contract loader.
 */

/**
 * Log levels controlling the verbosity of output.
 *
 * Maps to standard severity levels. The cycle runner sets the level
 * from CLI options; default is "info".
 */
export type LogLevel = "debug" | "info" | "warn" | "error";

/**
 * Numeric severity mapping for log level comparison.
 *
 * Higher values = more severe. Used by shouldLog() to filter messages.
 */
const LOG_LEVEL_VALUES: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

/**
 * Maps Python runtime phase prefixes to TypeScript phase names.
 *
 * Preserves the bracket-prefixed output format used in ciclo.py:
 * [perceber], [planejar], [circuit_breaker], [regras], [agir], [avaliar].
 */
type PhaseName = "perceber" | "planejar" | "circuit_breaker" | "regras" | "agir" | "avaliar";

/**
 * Structured logger instance with phase-aware formatting.
 *
 * Supports phase prefixes and log level filtering. Each method
 * (debug, info, warn, error) checks the current level before writing.
 * Output goes to stdout for info/debug and stderr for warn/error,
 * matching Node.js conventions for pipe and redirect behavior.
 *
 * Used by: All domain modules that need runtime observability.
 */
export class Logger {
  /** Current minimum log level for output filtering. */
  private level: LogLevel;

  /** Optional phase prefix prepended to all messages in this logger instance. */
  private phase: PhaseName | undefined;

  /**
   * Creates a Logger instance.
   *
   * @param level - Minimum log level to output (default: "info").
   * @param phase - Optional phase name for bracket-prefix formatting.
   */
  constructor(level: LogLevel = "info", phase?: PhaseName) {
    this.level = level;
    this.phase = phase;
  }

  /**
   * Creates a child logger with a specific phase prefix.
   *
   * The child inherits the parent's log level but adds a phase
   * prefix to all messages. This mirrors the Python runtime's
   * [perceber], [planejar], etc. conventions.
   *
   * @param phase - The phase name to use as prefix.
   * @returns A new Logger instance with the phase prefix set.
   *
   * Used by: Cycle runner to create phase-specific loggers.
   */
  child(phase: PhaseName): Logger {
    return new Logger(this.level, phase);
  }

  /**
   * Sets the minimum log level for this logger instance.
   *
   * @param level - The new minimum log level.
   */
  setLevel(level: LogLevel): void {
    this.level = level;
  }

  /**
   * Logs a debug message (phase: debug).
   *
   * @param message - The message to log.
   * @param data - Optional structured data to include.
   */
  debug(message: string, data?: Record<string, unknown>): void {
    if (this.shouldLog("debug")) {
      this.write("debug", message, data);
    }
  }

  /**
   * Logs an info message (phase: info).
   *
   * @param message - The message to log.
   * @param data - Optional structured data to include.
   */
  info(message: string, data?: Record<string, unknown>): void {
    if (this.shouldLog("info")) {
      this.write("info", message, data);
    }
  }

  /**
   * Logs a warning message (phase: warn).
   *
   * @param message - The message to log.
   * @param data - Optional structured data to include.
   */
  warn(message: string, data?: Record<string, unknown>): void {
    if (this.shouldLog("warn")) {
      this.write("warn", message, data);
    }
  }

  /**
   * Logs an error message (phase: error).
   *
   * @param message - The message to log.
   * @param data - Optional structured data to include.
   */
  error(message: string, data?: Record<string, unknown>): void {
    if (this.shouldLog("error")) {
      this.write("error", message, data);
    }
  }

  /**
   * Determines whether a message at the given level should be output.
   *
   * @param msgLevel - The severity of the message to check.
   * @returns True if the message level meets or exceeds the configured threshold.
   */
  private shouldLog(msgLevel: LogLevel): boolean {
    return LOG_LEVEL_VALUES[msgLevel] >= LOG_LEVEL_VALUES[this.level];
  }

  /**
   * Formats and writes a log message to the appropriate stream.
   *
   * Format: [phase] message  or  message  (when no phase is set).
   * Data objects are appended as JSON after a pipe separator.
   * Writes to stderr for warn/error levels; stdout otherwise.
   *
   * @param level - The log level of the message.
   * @param message - The message text.
   * @param data - Optional structured data.
   */
  private write(level: LogLevel, message: string, data?: Record<string, unknown>): void {
    const prefix = this.phase ? `[${this.phase}] ` : "";
    let line = `${prefix}${message}`;
    if (data && Object.keys(data).length > 0) {
      line += ` | ${JSON.stringify(data)}`;
    }

    const stream = level === "warn" || level === "error" ? process.stderr : process.stdout;
    stream.write(line + "\n");
  }
}

/**
 * Singleton root logger instance.
 *
 * Shared across the runtime for general logging. Phase-specific loggers
 * should be created via logger.child("phaseName") for per-phase output.
 *
 * Used by: Contract loader, state manager, any module needing general logging.
 */
export const logger = new Logger("info");
