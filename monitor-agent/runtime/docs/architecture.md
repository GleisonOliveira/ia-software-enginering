# Architecture -- @monitor-agent/runtime

## System Overview

The monitor-agent runtime executes AI agents through a **perceive-plan-act-evaluate** cycle. Agent behavior is defined by 9 YAML contract files (embedded in `.md` files), and the runtime interprets these contracts to orchestrate LLM-driven tool execution with safety guards, telemetry, and trace persistence.

The system is a TypeScript ESM rewrite of the original Python runtime, preserving Portuguese field names in contracts and output for backward compatibility.

## Execution Cycle

Every agent execution follows the perceive-plan-act-evaluate loop:

```
                  +-----------+
                  |  Contracts |
                  |  (YAML)   |
                  +-----+-----+
                        |
                        v
               +-----------------+
               |   ContractLoader |
               |   (9 contracts)  |
               +--------+--------+
                        |
                        v
               +-----------------+
               |  StateManager   |
               |  (AgentState)   |
               +--------+--------+
                        |
    +-------------------+-------------------+
    |                   |                   |
    v                   v                   v
+--------+        +---------+         +----------+
| Tools  |        |  Rules  |         |  Memory  |
| Build  |        | Limits  |         |  Config  |
+---+----+        +----+----+         +----+-----+
    |                  |                   |
    +------------------+-------------------+
                       |
                       v
            +---------------------+
            |    CycleRunner      |
            |  (main loop)        |
            +----+----+----+-----+
                 |    |    |    |
     +-----------+    |    |    +-----------+
     |                |    |                |
     v                v    v                v
+----------+   +---------+  +---------+  +----------+
| PERCEBER |   | PLANEJAR|  |  AGIR   |  | AVALIAR  |
| (perceive)|  | (plan)  |  | (act)   |  | (evaluate)|
+----+-----+   +----+----+  +----+----+  +-----+----+
     |              |            |             |
     v              v            v             v
  Perception    Planner      ToolExecutor   Evaluator
  Builder       (LLM)        CircuitBreak   PayloadValidator
                               Breaker
     |              |            |             |
     +--------------+------------+-------------+
                        |
                        v
               +------------------+
               |    Telemetry     |
               |    (metrics)     |
               +--------+---------+
                        |
                        v
               +------------------+
               |    trace.json    |
               |    (persist)     |
               +------------------+
```

## Domain Descriptions

### shared

**Files:** `src/shared/logger.ts`, `src/shared/env.ts`, `src/shared/shared.types.ts`

The shared domain provides cross-cutting utilities used by all other domains.

- **`Logger`** -- Structured, phase-aware logger. Supports log levels (`debug`, `info`, `warn`, `error`), phase prefixes (`[perceber]`, `[planejar]`, etc.), and child loggers for phase-specific output. Writes to stdout for info/debug and stderr for warn/error.
- **`EnvLoader`** -- Loads `.env` files via dotenv and validates environment variables using Zod schemas. Provides provider-specific default base URLs for LLM providers. Caches the validated result and supports cache reset for testing.
- **Shared types** -- `AgentType`, `ParamType`, `ActionType`, `QualityRating`, `HookAction`, `OutputFormat`, `TokenUsage`, `ActionResult`, `SkillParamSchema`.

### contracts

**Files:** `src/contracts/loader.ts`, `src/contracts/schemas.ts`, `src/contracts/contracts.types.ts`

The contracts domain loads agent definitions from `.md` files containing fenced YAML code blocks.

- **`ContractLoader`** -- Extracts YAML from 9 contract files (agente, ciclo, planejador, caixa_ferramentas, executor, regras, ganchos, habilidades, memoria). Validates against composite Zod schema. Handles missing files and malformed YAML gracefully.
- **`schemas.ts`** -- 9 Zod schemas (`AgentContractSchema`, `LoopContractSchema`, `PlannerContractSchema`, `ToolboxContractSchema`, `ExecutorContractSchema`, `RulesContractSchema`, `HooksContractSchema`, `SkillsContractSchema`, `MemoryContractSchema`) plus the composite `AllContractsSchema`.
- **Contract types** -- All types derived from Zod schemas via `z.infer`: `AgentContract`, `LoopContract`, `PlannerContract`, `ToolboxContract`, `ExecutorContract`, `RulesContract`, `HooksContract`, `SkillsContract`, `MemoryContract`, `AllContracts`.

### core

**Files:** `src/core/state.ts`, `src/core/cycle.ts`, `src/core/state.types.ts`, `src/core/cycle.types.ts`

The core domain manages agent state and the main execution loop.

- **`StateManager`** -- Creates initial `AgentState` from loaded contracts and user input. Extracts limits from `regras.md`, applies CLI overrides for agent type, initializes all mutable tracking fields to zero values.
- **`CycleRunner`** -- Central orchestrator that ties all domains together. Manages the perceive-plan-act-evaluate loop, enforces resource limits (steps, tokens, time, per-tool calls), handles stagnation detection, circuit breaker fallback, mandatory tool enforcement, lifecycle hooks, KPI panel display, and trace persistence. Receives 11 dependencies via constructor injection.
- **State types** -- `AgentState` (immutable public interface), `MutableState` (internal mutation wrapper), `StateLimits` (resource limits from contracts).
- **Cycle types** -- `CycleConfig` (execution parameters), `HistoryEntry` (per-step record), `Evaluation` (objective check), `Trace` (full execution log), `ValidActionType`.

### llm

**Files:** `src/llm/llm-config.ts`, `src/llm/llm-client.ts`, `src/llm/provider-factory.ts`, `src/llm/structured-output.ts`, `src/llm/llm.types.ts`

The LLM domain provides a provider-agnostic abstraction over the Vercel AI SDK.

- **`LlmConfigResolver`** -- Resolves `LlmConfig` from validated environment variables. Maps provider identifiers to API key env vars (e.g., `OPENAI_API_KEY`). Validates configuration and returns user-friendly error messages for missing fields.
- **`LlmClient`** -- Wraps the AI SDK's `generateText()` with injected configuration. Provides `callLlm()` for text generation with token usage extraction and error normalization.
- **`ProviderFactory`** -- Creates AI SDK `LanguageModel` instances from `LlmConfig`. Maps provider strings (`openai`, `anthropic`, `google`, `mistral`, `openrouter`) to their corresponding `@ai-sdk/*` factory functions. OpenRouter uses the OpenAI SDK with a custom base URL.
- **`StructuredOutputHandler`** -- Generates validated structured output using `generateObject()` with Zod schema constraints. Includes retry logic (up to 3 attempts) with validation error feedback for self-correction.
- **LLM types** -- `LlmProvider`, `LlmConfig`, `CallLlmOptions`, `LlmResponse`, `LlmUsage`, `StructuredOutputOptions<T>`.

### executor

**Files:** `src/executor/executor.ts`, `src/executor/circuit-breaker.ts`, `src/executor/evaluator.ts`, `src/executor/payload-validator.ts`, `src/executor/executor.types.ts`

The executor domain handles tool execution, validation, and result evaluation.

- **`ToolExecutor`** -- Executes registered tools by name with optional payload validation and retry logic. Returns `ExecutionResult` with `ActionResult` and token usage tracking.
- **`CircuitBreaker`** -- Validates LLM plans against available tools and contract rules before execution. Detects invalid action types, missing tool names, nonexistent tools, and policy violations. Applies auto-correction (e.g., case-insensitive tool name matching).
- **`Evaluator`** -- Post-action evaluation of tool execution results. Checks the plan's action type, tool execution success, and output schema compliance. Assigns quality ratings (`completa`, `parcial`, `falha`) and determines objective completion.
- **`PayloadValidator`** -- Validates tool input arguments against skill contract schemas before execution. Also validates tool output data post-execution. Returns error lists rather than throwing.
- **Executor types** -- `ValidationResult`, `ExecutionResult`, `ToolExecutorContext`, `ToolInputSchema`.

### tools

**Files:** `src/tools/tool-registry.ts`, `src/tools/tool-builder.ts`, `src/tools/hooks.ts`, `src/tools/tools.types.ts`

The tools domain manages the registry of executable tools and lifecycle hooks.

- **`ToolRegistry`** -- Maintains a `Map<string, ToolRegistryEntry>` for O(1) tool lookup by name. Provides methods for registration, lookup, enumeration, and conversion to `Map<string, ToolFunction>` for the executor.
- **`ToolBuilder`** -- Constructs executable `ToolFunction` instances from skill contract definitions. Each tool attempts LLM-backed data generation first, falling back to mock data when no API key is available. Creates `ToolRegistryEntry` objects for registration.
- **`HookExecutor`** -- Executes lifecycle hooks declared in `ganchos.md` at specific cycle points: `antes_da_etapa`, `apos_etapa`, `antes_da_acao`, `apos_acao`, `em_erro`. Supports `log` (structured logger) and `alerta` (stderr alert) action types.
- **Tool types** -- `ToolFunction`, `ToolDefinition`, `ToolRegistryEntry`.

### planner

**Files:** `src/planner/planner.ts`, `src/planner/prompt-builder.ts`, `src/planner/perception.ts`, `src/planner/planner.types.ts`

The planner domain generates the LLM's next action decision.

- **`PerceptionBuilder`** -- Constructs the perception prompt (user prompt for the LLM) from `AgentState`. Includes user input, agent mode, event context, history with tool results, tools used, step/token counts, and stagnation warnings.
- **`PromptBuilder`** -- Constructs the system prompt from all contract definitions. Assembles agent identity, tool descriptions with schemas, response format specification, planner rules, agent policies, and mode-specific instructions (interactive, goal-oriented, autonomous).
- **`Planner`** -- Combines perception and system prompt to call the LLM for structured plan generation via `StructuredOutputHandler`. Falls back to a mock planner when no API key is available, cycling through tools in order. Normalizes raw LLM output (`snake_case`) to `Plan` interface (`camelCase`).
- **Planner types** -- `Plan` (structured LLM output), `Perception` (string alias), `PlannerContext`.

### telemetry

**Files:** `src/telemetry/telemetry.ts`, `src/telemetry/telemetry.types.ts`

The telemetry domain collects execution metrics for debugging, audit, and the KPI panel.

- **`Telemetry`** -- Collects timestamped events, per-phase timing (via `PhaseMarker`), token usage aggregation, tool success/failure counters, circuit breaker activation counts, and payload validation failure counts. Generates health metrics (`HealthMetrics`), performance data (`PerformanceData`), and a complete summary (`TelemetrySummary`) for trace persistence.
- **Telemetry types** -- `TelemetryEvent`, `TelemetryEventType`, `PhaseName`, `PhaseMarker`, `PhaseStats`, `HealthMetrics`, `PerformanceData`, `TelemetrySummary`.

### cli

**Files:** `src/cli/index.ts`, `src/cli/cli.types.ts`, `src/cli/commands/run.ts`, `src/cli/commands/validate.ts`, `src/cli/commands/trace.ts`, `src/cli/commands/analyze.ts`, `src/cli/commands/replay.ts`

The CLI domain provides the `monitor-runtime` binary entry point via Commander.js.

- **`CliApp`** -- Wraps Commander.js and registers all 5 subcommands. Receives all command handlers via dependency injection.
- **`RunCommand`** -- Executes an agent with `--agente` (path) and `--entrada` (input text) options. Delegates to `CycleRunner.run()`.
- **`ValidateCommand`** -- Validates all 9 contracts from an agent directory. Reports success or lists validation errors.
- **`TraceCommand`** -- Reads and displays `trace.json` with step-by-step execution history, health metrics, performance data, and summary.
- **`AnalyzeCommand`** -- Reads the last trace, builds a compact summary, runs it through an analyzer agent via `CycleRunner`, and generates a markdown report (`analise-agente.md`).
- **`ReplayCommand`** -- Reads the last trace to extract input parameters and re-runs the agent with the same configuration.
- **CLI types** -- `RunOptions`, `ValidateOptions`, `TraceOptions`, `AnalyzeOptions`, `ReplayOptions`.

## Data Flow

```
CLI Input
  |
  v
CliApp.parse(argv)
  |
  v
RunCommand.execute(RunOptions)
  |
  v
CycleRunner.run(CycleConfig)
  |
  +---> ContractLoader.loadAllContracts(agentPath)
  |       |
  |       v
  |     AllContracts (9 validated contracts)
  |
  +---> StateManager.createState(contracts, input)
  |       |
  |       v
  |     AgentState
  |
  +---> ToolBuilder.buildEntry(skill) x N
  |       |
  |       v
  |     ToolRegistry.registerAll(entries)
  |
  +---> Telemetry(agent, type)
  |
  +---> [LOOP while !concluido && etapa < maxEtapas]
          |
          +---> PerceptionBuilder.build(state)
          |       |
          |       v
          |     perception (string)
          |
          +---> Planner.plan(state, contracts)
          |       |
          |       +---> PromptBuilder.build(contracts) -> systemPrompt
          |       +---> StructuredOutputHandler.generate(schema, prompts)
          |       |       |
          |       |       +---> ProviderFactory.create(config) -> LanguageModel
          |       |       +---> generateObject() via AI SDK
          |       |       |
          |       |       v
          |       |     Plan (proximaAcao, nomeFerramenta, args, criterioSucesso)
          |       |
          |       v
          |     { plan, tokens }
          |
          +---> CircuitBreaker.validate(plan, contracts)
          |       |
          |       v
          |     ValidationResult { valido, erros }
          |       |
          |       +---> autoCorrect(plan, contracts) if invalid
          |
          +---> [if CHAMAR_FERRAMENTA]
          |       |
          |       +---> PayloadValidator.validate(toolName, args, contracts)
          |       +---> ToolExecutor.execute(toolName, args, toolMap, contracts)
          |       |       |
          |       |       v
          |       |     ExecutionResult { resultado, tokensUsados }
          |       |
          |       v
          |     ActionResult { sucesso, dados, erro }
          |
          +---> Evaluator.evaluate(plan, actionResult, contracts)
          |       |
          |       v
          |     Evaluation { objetivoAlcancado, motivo, qualidade }
          |
          +---> Telemetry.registrar(...)
          +---> HookExecutor.execute(hookName, contracts, params)
          |
          v
        [END LOOP]
  |
  +---> Telemetry.resumoCompleto()
  +---> CycleRunner.buildTrace(...)
  +---> fs.writeFileSync("trace.json", trace)
  |
  v
KPI Panel Output -> stdout
```

## Dependency Injection Diagram

All dependencies flow through constructors. The `CycleRunner` is the composition root within the runtime (not the CLI entry point, which injects it into command handlers).

```
CliApp
  |-- RunCommand
  |     |-- CycleRunner
  |           |-- ContractLoader
  |           |     |-- Logger
  |           |-- StateManager
  |           |-- PerceptionBuilder
  |           |-- Planner
  |           |     |-- StructuredOutputHandler
  |           |     |     |-- LlmConfigResolver
  |           |     |     |     |-- EnvLoader
  |           |     |     |-- ProviderFactory
  |           |     |     |-- Logger
  |           |     |-- PerceptionBuilder
  |           |     |-- PromptBuilder
  |           |     |-- Logger
  |           |-- CircuitBreaker
  |           |-- ToolExecutor
  |           |     |-- PayloadValidator
  |           |     |-- Logger
  |           |-- Evaluator
  |           |     |-- PayloadValidator
  |           |-- ToolRegistry
  |           |-- ToolBuilder
  |           |     |-- Logger
  |           |-- HookExecutor
  |           |     |-- Logger
  |           |-- Logger
  |-- ValidateCommand
  |     |-- ContractLoader
  |     |-- Logger
  |-- TraceCommand
  |     |-- Logger
  |-- AnalyzeCommand
  |     |-- CycleRunner (same tree as RunCommand)
  |     |-- Logger
  |-- ReplayCommand
        |-- CycleRunner (same tree as RunCommand)
        |-- Logger
```

## Trace File Format

The `trace.json` file written after each execution contains the complete `Trace` interface:

```json
{
  "traceId": "a1b2c3d4e5f6",
  "tipoAgente": "task_based",
  "entrada": "Alerta de latencia no servico X",
  "evento": "cpu_above_90_percent",
  "tempoTotalSegundos": 12.5,
  "tokensConsumidos": {
    "prompt": 1500,
    "completion": 800,
    "total": 2300
  },
  "etapas": [
    {
      "etapa": 1,
      "percepcao": "Alerta: ...\nModo: task_based\n...",
      "plano": {
        "proximaAcao": "CHAMAR_FERRAMENTA",
        "nomeFerramenta": "buscar_metricas",
        "argumentosFerramenta": { "servico": "api-gateway" },
        "criterioSucesso": "Metricas coletadas com sucesso"
      },
      "resultadoAcao": {
        "sucesso": true,
        "dados": { "cpu_avg": 85.2, "memoria": "4.2GB" },
        "erro": "",
        "_tokens": { "prompt": 0, "completion": 0, "total": 0 },
        "_entrada": { "servico": "api-gateway" }
      },
      "avaliacao": {
        "objetivoAlcancado": false,
        "motivo": "Step OK -- continue",
        "qualidade": "completa",
        "problemasSaida": []
      }
    }
  ],
  "resumo": "Objetivo: Diagnostico de latencia\nEtapas executadas: 3\n...",
  "agente": "my-agent",
  "telemetryStream": [ ... ],
  "auditLogs": [ ... ],
  "healthMetrics": {
    "traceId": "a1b2c3d4e5f6",
    "taxaSucessoFerramentas": 100,
    "ferramentasSucesso": 3,
    "ferramentasFalha": 0,
    "circuitBreakerAtivacoes": 1,
    "validacaoPayloadFalhas": 0,
    "chamadasLlm": 3
  },
  "performanceData": {
    "traceId": "a1b2c3d4e5f6",
    "tempoTotalMs": 12500,
    "tokens": { "prompt": 1500, "completion": 800, "total": 2300 },
    "chamadasLlm": 3,
    "fases": {
      "perceber": { "totalMs": 50, "contagem": 3, "maxMs": 25, "mediaMs": 16.7 },
      "planejar": { "totalMs": 8000, "contagem": 3, "maxMs": 3500, "mediaMs": 2666.7 },
      "agir": { "totalMs": 2000, "contagem": 3, "maxMs": 1000, "mediaMs": 666.7 },
      "avaliar": { "totalMs": 100, "contagem": 3, "maxMs": 50, "mediaMs": 33.3 }
    }
  }
}
```

## Key Design Decisions

1. **Contract-driven execution:** Agent behavior is fully defined by 9 YAML contracts. The runtime interprets these contracts to determine tools, rules, limits, hooks, and output format.

2. **Portuguese field names preserved:** Contract-facing fields use Portuguese names (`sucesso`, `dados`, `proxima_acao`) for backward compatibility with the Python runtime.

3. **Mock planner fallback:** When no LLM API key is available, the planner cycles through tools in order, enabling testing and development without credentials.

4. **Circuit breaker with auto-correction:** Invalid LLM plans are caught before execution and auto-corrected when possible (e.g., case-insensitive tool name matching), with fallback to unused tools.

5. **Immutable public interface:** `AgentState` uses `readonly` fields. Internal mutations use `MutableState` wrapper, preventing accidental state corruption.

6. **Trace-first observability:** Every execution produces a complete `trace.json` with telemetry events, audit logs, health metrics, and performance data.

## Cross-References

- **Project overview:** See [AGENTS.md](../AGENTS.md)
- **Coding conventions:** See [STYLEGUIDE.md](../STYLEGUIDE.md)
- **Testing guidelines:** See [TESTING.md](../TESTING.md)
