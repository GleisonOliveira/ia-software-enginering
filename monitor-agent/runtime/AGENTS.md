# AGENTS.md -- @monitor-agent/runtime

## Project

**Name:** `@monitor-agent/runtime`
**Description:** AI agent execution runtime -- multi-provider LLM with structured outputs.
**Version:** 1.0.0
**Node:** >= 20.0.0
**Module system:** ESM (`"type": "module"`)

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | TypeScript 5.8+, strict mode |
| Runtime | Node.js >= 20 (ES2022 target) |
| Module resolution | NodeNext |
| LLM integration | Vercel AI SDK 7+ (`ai` package) |
| Providers | `@ai-sdk/openai`, `@ai-sdk/anthropic`, `@ai-sdk/google`, `@ai-sdk/mistral` |
| Validation | Zod 3.25+ (runtime + type inference) |
| YAML parsing | js-yaml |
| CLI | Commander.js 12+ |
| Testing | Jest 29+ with ts-jest ESM preset |
| Linting | ESLint 9+ with typescript-eslint |
| Formatting | Prettier |

## Architecture Overview

The runtime follows a **perceive-plan-act-evaluate** cycle organized into 9 domain modules. Each domain contains classes (not standalone functions), types (`*.types.ts`), and optional barrel exports (`index.ts`). Dependencies are injected via constructor parameters -- no module-level singletons.

### Domains

| Domain | Purpose | Key Classes |
|--------|---------|-------------|
| **shared** | Cross-cutting: logging, environment config | `Logger`, `EnvLoader` |
| **contracts** | YAML contract loading from .md files | `ContractLoader` |
| **core** | State management, main cycle orchestration | `StateManager`, `CycleRunner` |
| **llm** | Provider-agnostic LLM abstraction | `LlmConfigResolver`, `LlmClient`, `ProviderFactory`, `StructuredOutputHandler` |
| **executor** | Tool execution, validation, evaluation | `ToolExecutor`, `CircuitBreaker`, `Evaluator`, `PayloadValidator` |
| **tools** | Tool registry, builder, lifecycle hooks | `ToolRegistry`, `ToolBuilder`, `HookExecutor` |
| **planner** | Perception building, prompt assembly, planning | `PerceptionBuilder`, `PromptBuilder`, `Planner` |
| **telemetry** | Execution metrics, timing, health tracking | `Telemetry` |
| **cli** | Commander.js entry point and subcommands | `CliApp`, `RunCommand`, `ValidateCommand`, `TraceCommand`, `AnalyzeCommand`, `ReplayCommand` |

### Supporting

| Path | Purpose |
|------|---------|
| `src/types/index.ts` | Public barrel export for all types and Zod schemas |
| `src/llm/llm.types.ts` | LLM provider interfaces |
| `src/core/state.types.ts` | Agent state and limits interfaces |
| `src/core/cycle.types.ts` | Cycle config, history, trace, evaluation interfaces |
| `src/executor/executor.types.ts` | Execution result and validation interfaces |
| `src/tools/tools.types.ts` | Tool function and definition interfaces |
| `src/planner/planner.types.ts` | Plan and perception interfaces |
| `src/telemetry/telemetry.types.ts` | Telemetry event, phase, health, and performance interfaces |
| `src/cli/cli.types.ts` | CLI option interfaces |

## Directory Structure

```
runtime/
  src/
    shared/         -- Logger, EnvLoader, shared.types.ts
    contracts/      -- ContractLoader, schemas.ts, contracts.types.ts
    core/           -- StateManager, CycleRunner, state.types.ts, cycle.types.ts
    llm/            -- LlmConfigResolver, LlmClient, ProviderFactory, StructuredOutputHandler, llm.types.ts
    executor/       -- ToolExecutor, CircuitBreaker, Evaluator, PayloadValidator, executor.types.ts
    tools/          -- ToolRegistry, ToolBuilder, HookExecutor, tools.types.ts
    planner/        -- PerceptionBuilder, PromptBuilder, Planner, planner.types.ts
    telemetry/      -- Telemetry, telemetry.types.ts
    cli/            -- CliApp, cli.types.ts
      commands/     -- RunCommand, ValidateCommand, TraceCommand, AnalyzeCommand, ReplayCommand
    types/          -- index.ts (barrel export)
  tests/
    shared/         -- logger.test.ts, env.test.ts
    contracts/      -- loader.test.ts
    core/           -- state.test.ts, cycle.test.ts
    llm/            -- llm-config.test.ts, llm-client.test.ts, provider-factory.test.ts, structured-output.test.ts
    executor/       -- executor.test.ts, circuit-breaker.test.ts, evaluator.test.ts, payload-validator.test.ts
    tools/          -- tool-registry.test.ts, tool-builder.test.ts, hooks.test.ts
    planner/        -- planner.test.ts, perception.test.ts, prompt-builder.test.ts
    telemetry/      -- telemetry.test.ts
    cli/            -- index.test.ts
      commands/     -- run.test.ts, validate.test.ts, trace.test.ts, analyze.test.ts, replay.test.ts
  docs/
    architecture.md -- Detailed architecture documentation
  AGENTS.md         -- This file
  STYLEGUIDE.md     -- Coding conventions
  TESTING.md        -- Testing guidelines
  package.json
  tsconfig.json
  eslint.config.mjs
  jest.config.mjs
```

## Key Design Decisions

1. **OOP with classes:** Every module is a class. No standalone exported functions except type-level utilities and constants. Classes enable constructor-based DI and clear ownership of state.

2. **Strict dependency injection:** All dependencies flow through constructors. `CycleRunner` receives 11 injected dependencies. No module-level singletons or global state.

3. **Zod schemas as source of truth:** Contract validation schemas (`contracts/schemas.ts`) define both runtime validation rules and TypeScript types via `z.infer`. Changing a schema automatically updates the corresponding type.

4. **ESM throughout:** All imports use `.js` extensions (NodeNext resolution). The project uses `"type": "module"` in package.json.

5. **Test-per-file:** Every non-barrel, non-type source file has a corresponding test file in `tests/<domain>/`. Type files (`*.types.ts`) and barrel files (`index.ts`) are exempt.

6. **Contract field names in Portuguese:** The runtime preserves Portuguese field names (`sucesso`, `dados`, `erro`, `proxima_acao`, `nome_ferramenta`) for backward compatibility with the Python runtime and existing agent definitions.

7. **Provider-agnostic LLM layer:** The Vercel AI SDK wraps all providers behind a unified `LanguageModel` interface. OpenRouter uses the OpenAI SDK with a custom baseURL.

8. **Mock planner fallback:** When no API key is available, the planner cycles through available tools in order, enabling development and testing without credentials.

9. **Trace persistence:** Every execution writes a `trace.json` containing the full telemetry stream, audit logs, health metrics, and performance data for replay and analysis.

## Cross-References

- **Coding conventions:** See [STYLEGUIDE.md](./STYLEGUIDE.md)
- **Testing guidelines:** See [TESTING.md](./TESTING.md)
- **Detailed architecture:** See [docs/architecture.md](./docs/architecture.md)
