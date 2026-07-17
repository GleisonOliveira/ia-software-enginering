# Implementation Plan — Runtime TypeScript

> **Status:** IN PROGRESS
> **Created:** 2026-07-15
> **Target:** `monitor-agent/runtime/`
> **Last Updated:** 2026-07-17

## Overview

Conversion of the Python AI agent execution runtime to TypeScript with:

- Domain-based architecture
- Zod schema validation
- Dependency injection via classes
- Strong typing (no `any`)
- Jest unit tests (one test file per source file)
- ESLint + Prettier
- Node 20+ compatibility

### OOP Requirement (Strict)

**All runtime logic must be implemented inside classes.** Standalone exported functions are forbidden.

- Every domain module (executor, planner, tools, llm, etc.) must export a **class**, not standalone functions.
- Pure utility logic (validators, builders, formatters) must be private or static methods inside a class — never exported as top-level functions.
- Classes must receive dependencies via **constructor injection**, not via module-level singletons or global state.
- Each class must have a single, well-defined responsibility (Single Responsibility Principle).
- The only exceptions are:
  - `*.types.ts` files (type-only, no runtime logic)
  - Zod schema definitions in `schemas.ts` (constant exports, not functions)
  - `src/types/index.ts` (re-export barrel)

### Test-per-File Requirement

**Every `.ts` source file must have a corresponding `.test.ts` file.** No exceptions (excluding type-only files).

- For each `src/<domain>/<name>.ts` → there must be `tests/<domain>/<name>.test.ts`
- Type-only files (`*.types.ts`) and barrel re-exports (`index.ts`) are exempt.
- Zod schema files (`schemas.ts`) must have tests verifying schema validation behavior.
- All external dependencies (LLM providers, filesystem, network) must be mocked in tests.
- Tests run via `npm test` (Jest with `--experimental-vm-modules` for ESM).

### Key Architectural Decision: Multi-Provider LLM Support

The Python runtime was hardcoded to OpenAI (`openai` SDK + `gpt-4o-mini`). The TypeScript version uses **Vercel AI SDK** (`ai` package) as the provider abstraction layer, making the LLM provider, model, and token limits fully interchangeable.

**Why Vercel AI SDK:**

- Unified API across OpenAI, Anthropic, Google Gemini, Mistral, OpenRouter, and 20+ providers
- Native Zod schema support for structured outputs (`Output.object()`)
- Provider-specific structured output translation handled automatically
- No vendor lock-in — change provider via env var + model string
- Active maintenance and TypeScript-first design
- OpenRouter supported via `@ai-sdk/openai` with custom `baseURL` (no extra dependency, Node 20+ compatible)

**Structured Outputs Compatibility:**

- OpenAI: `response_format.json_schema` (constrained decoding, 100% schema adherence)
- Anthropic: `output_config.format` (grammar-based, ~99%+ reliability)
- Google Gemini: `response_schema` + `response_mime_type` (constrained decoding)
- AI SDK normalizes all of these behind a single `Output.object({ schema })` API

---

## Code Commenting Conventions

All TypeScript code in this project **must** include comments explaining **what** the code does and **why** it is being used. This applies to:

### Module-level comments

- Every `.ts` file must begin with a block comment (`/** ... */`) describing the module's purpose and responsibility.
- Include which domain it belongs to (core, llm, executor, tools, planner, telemetry, cli).

### Class/method/type comments

- Every exported class must have a JSDoc comment explaining:
  - **What** it does (brief description)
  - **Why** this class exists and was designed this way
  - **Where** it is used and in what context (e.g., "Used by: cycle runner, planner")
- Every constructor must document its injected dependencies via `@param` tags.
- Every public method must have a JSDoc comment with:
  - **What** it does (brief description)
  - **Why** this approach was chosen (design rationale when non-obvious)
  - `@param` tags for all parameters
  - `@returns` description
- For type aliases and interfaces, document each field with a brief inline comment if the name alone is not self-explanatory.

### Inline comments

- Use inline comments (`//`) to explain non-obvious logic, workarounds, or domain-specific rules.
- Do **not** restate what the code does — focus on intent and reasoning.
- Comment complex business rules (circuit breaker logic, stagnation detection, token limits) to connect the code back to the contracts.

### Why comments are mandatory

- The Python runtime used Portuguese docstrings extensively for this purpose. The TypeScript version must maintain the same level of clarity.
- The agent contracts (YAML `.md` files) define _what_ the system should do; comments in code explain _how_ and _why_ it is implemented that way.
- Future maintainers and AI agents reading the code need context that type signatures alone do not convey.

### Example (Class-Based)

```typescript
/**
 * Validates LLM responses against available tools before execution.
 *
 * Why: The LLM may return invalid tool names or malformed arguments.
 * This circuit breaker prevents runtime errors and enables auto-correction.
 * Used by: CycleRunner, Planner
 */
class CircuitBreaker {
  /**
   * @param availableTools - Set of tool names currently registered in the ToolRegistry
   */
  constructor(private readonly availableTools: Set<string>) {}

  /**
   * Validates a plan and applies auto-correction when possible.
   * @param plan - The structured plan returned by the LLM
   * @returns List of validation problems (empty = valid)
   */
  validate(plan: Plan): string[] {
    // Check if proxima_acao is one of the valid action types defined in contracts
    const validActions = new Set(["CHAMAR_FERRAMENTA", "FINALIZAR", "PERGUNTAR_USUARIO"]);
    // ...
  }
}
```

---

## Phase 1 — Project Setup

### Acceptance Criteria

- [ ] `tsc --noEmit` passes with zero errors
- [ ] `npm run lint` passes with zero errors
- [ ] `npm test` passes (Jest setup verified)

### 1.1 Initialize package.json

- [x] Create `package.json` with all dependencies
- [x] Run `npm install`

### 1.2 TypeScript Configuration

- [x] Create `tsconfig.json` (strict mode, ES2022, moduleResolution bundler)
- [x] Verify `tsc --noEmit` passes

### 1.3 ESLint + Prettier

- [x] Create `eslint.config.mjs` with typescript-eslint rules
- [x] Create `.prettierrc` with consistent formatting
- [x] Add `lint` and `format` scripts to package.json
- [x] Verify `npm run lint` passes

### 1.4 Environment

- [x] Create `.env.example` (multi-provider keys + LLM_CONFIG defaults)
- [x] Create `.gitignore` (node_modules, dist, .env, trace.json, analise.json)

### 1.5 Jest

- [x] Create `jest.config.mjs` with ts-jest (ESM-compatible)
- [x] Add `test` script to package.json
- [x] Verify `npm test` passes

---

## Phase 2 — Types and Schemas

### Acceptance Criteria

- [x] `tsc --noEmit` passes with zero errors after all type files are created
- [x] `grep -r ": any" src/` returns zero results
- [x] All types are re-exported via `src/types/index.ts`

### 2.1 Shared Types

- [x] Create `src/shared/shared.types.ts` (TokenUsage, ActionResult, etc.)

### 2.2 LLM Provider Types

- [x] Create `src/llm/llm.types.ts` (LlmProvider, LlmConfig, StructuredOutputOptions, etc.)

### 2.3 Contract Schemas and Types

- [x] Create `src/contracts/schemas.ts` (Zod schemas for all contracts)
- [x] Create `src/contracts/contracts.types.ts` (types derived from Zod schemas)
- [x] Create `tests/contracts/schemas.test.ts` (verify schema validation behavior)

### 2.4 Core Types

- [x] Create `src/core/state.types.ts` (AgentState, StateLimits, etc.)
- [x] Create `src/core/cycle.types.ts` (CycleConfig, HistoryEntry, etc.)

### 2.5 Executor Types

- [x] Create `src/executor/executor.types.ts` (ExecutionResult, ValidationResult, etc.)

### 2.6 Tools Types

- [x] Create `src/tools/tools.types.ts` (ToolFunction, ToolDefinition, etc.)

### 2.7 Planner Types

- [x] Create `src/planner/planner.types.ts` (Plan, Perception, etc.)

### 2.8 Telemetry Types

- [x] Create `src/telemetry/telemetry.types.ts` (TelemetryEvent, PhaseMarker, etc.)

### 2.9 CLI Types

- [x] Create `src/cli/cli.types.ts` (RunOptions, ValidateOptions, etc.)

### 2.10 Public Re-exports

- [x] Create `src/types/index.ts` (public type re-exports)

---

## Phase 3 — Core Domain

### Acceptance Criteria

- [x] `ContractLoader.loadYamlFromMd()` extracts YAML from a test `.md` file correctly
- [x] `ContractLoader.loadAllContracts()` loads all 9 contracts from a valid agent_path
- [x] `StateManager.createState()` returns an `AgentState` with all required fields populated
- [x] `EnvConfig.load()` validates variables with Zod and returns defaults for `LLM_BASE_URL`
- [x] Unit tests with mocked provider pass
- [x] Every source file has a corresponding test file

### 3.1 Environment

- [x] Create `src/shared/env.ts` — `EnvConfig` class (dotenv loading with Zod validation)
- [x] Validate `LLM_BASE_URL` env var (provider-specific defaults when empty)
- [x] Create `tests/shared/env.test.ts`

### 3.2 Logger

- [x] Create `src/shared/logger.ts` — `Logger` class (structured logger replacing print statements)
- [x] Create `tests/shared/logger.test.ts`

### 3.3 Contract Loader

- [x] Create `src/contracts/loader.ts` — `ContractLoader` class (YAML extraction from .md files)
- [x] Implement `loadYamlFromMd(filePath)` method
- [x] Implement `loadAllContracts(agentPath)` method
- [x] Create `tests/contracts/loader.test.ts`

### 3.4 State Manager

- [x] Create `src/core/state.ts` — `StateManager` class (state creation from contracts)
- [x] Implement `createState(contracts, input, mode?, event?)` method
- [x] Create `tests/core/state.test.ts`

---

## Phase 4 — LLM Provider Abstraction

### Acceptance Criteria

- [x] `LlmClient.callLlm()` returns text and `TokenUsage` with configurable provider via env
- [x] `LlmConfig.resolveProvider()` selects the correct provider based on `LLM_PROVIDER`
- [x] `StructuredOutputHandler.generate()` validates LLM output against a Zod schema
- [x] Unit tests with mocked provider pass
- [x] Every source file has a corresponding test file

### 4.1 LLM Client (Provider-Agnostic)

- [x] Create `src/llm/llm-client.ts` — `LlmClient` class wrapping Vercel AI SDK `generateText`
- [x] Implement provider auto-detection from env vars
- [x] Implement `callLlm(options)` method with structured output support via `Output.object()`
- [x] Implement token usage extraction from AI SDK response
- [x] Create `tests/llm/llm-client.test.ts`

### 4.2 LLM Provider Config

- [x] Create `src/llm/llm-config.ts` — `LlmConfig` class
- [x] Implement `resolveProvider()` method — reads `LLM_PROVIDER` env, selects provider
- [x] Support provider-specific env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, etc.
- [x] Implement `LLM_BASE_URL` env var — custom endpoint for OpenRouter, local models, proxies
- [x] Implement model selection from `LLM_MODEL` env var
- [x] Implement token limits from `LLM_MAX_TOKENS` env var
- [x] Implement structured output schema conversion (Zod → provider format)
- [x] Create `tests/llm/llm-config.test.ts`

### 4.3 Structured Output Helpers

- [x] Create `src/llm/structured-output.ts` — `StructuredOutputHandler` class
- [x] Implement `generate<T>(schema, prompt, systemPrompt)` method using AI SDK `Output.object()`
- [x] Implement validation fallback for providers with lower structured output reliability
- [x] Implement retry logic with Zod validation error feedback
- [x] Create `tests/llm/structured-output.test.ts`

---

## Phase 5 — Executor Domain

### Acceptance Criteria

- [x] `PayloadValidator.validate()` returns error list for invalid args (empty list for valid args)
- [x] `CircuitBreaker.validate()` detects invalid plans and applies auto-correction
- [x] `ToolExecutor.execute()` executes a tool and returns `ActionResult`
- [x] `Evaluator.evaluate()` returns correct `objetivo_alcancado` and `qualidade`
- [x] Unit tests with mocked provider pass
- [x] Every source file has a corresponding test file

### 5.1 Payload Validator

- [x] Create `src/executor/payload-validator.ts` — `PayloadValidator` class
- [x] Implement `validate(toolName, args, contracts)` method using Zod
- [x] Create `tests/executor/payload-validator.test.ts`

### 5.2 Circuit Breaker

- [x] Create `src/executor/circuit-breaker.ts` — `CircuitBreaker` class
- [x] Implement `validate(plan)` method
- [x] Implement auto-correction and fallback logic
- [x] Create `tests/executor/circuit-breaker.test.ts`

### 5.3 Tool Executor

- [x] Create `src/executor/executor.ts` — `ToolExecutor` class
- [x] Implement `execute(toolName, args, tools, contracts)` method
- [x] Implement retry logic from contracts
- [x] Create `tests/executor/executor.test.ts`

### 5.4 Evaluator

- [x] Create `src/executor/evaluator.ts` — `Evaluator` class
- [x] Implement `evaluate(plan, actionResult, contracts)` method
- [x] Implement output validation against schema
- [x] Create `tests/executor/evaluator.test.ts`

---

## Phase 6 — Tools Domain

### Acceptance Criteria

- [x] `ToolBuilder.build()` returns an executable function from a skill definition
- [x] `ToolRegistry` allows lookup by tool name
- [x] `HookExecutor.execute()` fires configured hooks
- [x] Unit tests with mocked provider pass
- [x] Every source file has a corresponding test file

### 6.1 Tool Builder

- [x] Create `src/tools/tool-builder.ts` — `ToolBuilder` class
- [x] Implement `build(skill)` method
- [x] Implement LLM-based tool execution with mock fallback
- [x] Create `tests/tools/tool-builder.test.ts`

### 6.2 Tool Registry

- [x] Create `src/tools/tool-registry.ts` — `ToolRegistry` class
- [x] Implement tool lookup by name
- [x] Create `tests/tools/tool-registry.test.ts`

### 6.3 Hooks

- [x] Create `src/tools/hooks.ts` — `HookExecutor` class
- [x] Implement `execute(name, hookContract, params)` method
- [x] Create `tests/tools/hooks.test.ts`

---

## Phase 7 — Planner Domain

### Acceptance Criteria

- [ ] `PerceptionBuilder.build()` returns a string with entry, mode, and progress info
- [ ] `PromptBuilder.build()` includes instructions for all modes (interactive, goal_oriented, autonomous)
- [ ] `Planner.plan()` returns a valid `Plan` with correct `proxima_acao` type
- [ ] `Planner.mockPlan()` works without API key
- [ ] Unit tests with mocked provider pass
- [ ] Every source file has a corresponding test file

### 7.1 Perception Builder

- [ ] Create `src/planner/perception.ts` — `PerceptionBuilder` class
- [ ] Implement `build(state)` method
- [ ] Create `tests/planner/perception.test.ts`

### 7.2 Prompt Builder

- [ ] Create `src/planner/prompt-builder.ts` — `PromptBuilder` class
- [ ] Implement `build(contracts)` method
- [ ] Implement mode-specific instructions (interactive, goal_oriented, autonomous)
- [ ] Create `tests/planner/prompt-builder.test.ts`

### 7.3 Planner

- [ ] Create `src/planner/planner.ts` — `Planner` class
- [ ] Implement `plan(perception, contracts, history)` method using `LlmClient` with structured output
- [ ] Implement `mockPlan(perception, contracts, history)` fallback
- [ ] Create `tests/planner/planner.test.ts`

---

## Phase 8 — Telemetry Domain

### Acceptance Criteria

- [ ] `Telemetry.registerEvent()` records events with timestamp and trace_id
- [ ] `Telemetry.healthMetrics()` returns success rate and counters
- [ ] `Telemetry.performanceData()` returns phase timing with avg/max/total
- [ ] Unit tests with mocked provider pass
- [ ] Every source file has a corresponding test file

### 8.1 Telemetry Collector

- [ ] Create `src/telemetry/telemetry.ts` — `Telemetry` class
- [ ] Implement event registration, phase timing, token tracking
- [ ] Implement `healthMetrics()`, `performanceData()`, `auditLogs()` methods
- [ ] Create `tests/telemetry/telemetry.test.ts`

---

## Phase 9 — CLI

### Acceptance Criteria

- [ ] `RunCommand` executes an agent with `--agente` and `--entrada`
- [ ] `ValidateCommand` validates contracts and returns ok/failure
- [ ] `TraceCommand` displays the last trace.json
- [ ] `AnalyzeCommand` generates `analise-agente.md`
- [ ] `ReplayCommand` re-executes with the same input
- [ ] Unit tests with mocked provider pass
- [ ] Every source file has a corresponding test file

### 9.1 Entry Point

- [ ] Create `src/cli/index.ts` — `CliApp` class wrapping Commander.js
- [ ] Create `tests/cli/index.test.ts`

### 9.2 Commands

- [ ] Create `src/cli/commands/run.ts` — `RunCommand` class
- [ ] Create `tests/cli/run.test.ts`
- [ ] Create `src/cli/commands/validate.ts` — `ValidateCommand` class
- [ ] Create `tests/cli/validate.test.ts`
- [ ] Create `src/cli/commands/trace.ts` — `TraceCommand` class
- [ ] Create `tests/cli/trace.test.ts`
- [ ] Create `src/cli/commands/analyze.ts` — `AnalyzeCommand` class
- [ ] Create `tests/cli/analyze.test.ts`
- [ ] Create `src/cli/commands/replay.ts` — `ReplayCommand` class
- [ ] Create `tests/cli/replay.test.ts`

---

## Phase 10 — Core Cycle

### Acceptance Criteria

- [ ] `CycleRunner.run()` executes the full perceive→plan→act→evaluate cycle
- [ ] Time, token, and stagnation limits interrupt the cycle correctly
- [ ] `CycleRunner.replay()` re-executes with input from the previous trace
- [ ] `trace.json` is saved with all telemetry data
- [ ] KPI panel is displayed after each step
- [ ] Unit tests with mocked provider pass
- [ ] Every source file has a corresponding test file

### 10.1 Main Cycle

- [ ] Create `src/core/cycle.ts` — `CycleRunner` class
- [ ] Implement `run(agentPath, input, mode?, event?, output?)` method
- [ ] Implement `replay(agentPath)` method
- [ ] Implement `showTrace()` method
- [ ] Implement KPI panel display
- [ ] Implement stagnation detection
- [ ] Implement time/token limit checks
- [ ] Implement mandatory tool enforcement
- [ ] Implement sensitive action confirmation
- [ ] Implement trace file saving
- [ ] Create `tests/core/cycle.test.ts`

---

## Phase 11 — Documentation

### Acceptance Criteria

- [ ] `AGENTS.md`, `STYLEGUIDE.md`, `TESTING.md`, `docs/architecture.md` exist
- [ ] Each doc correctly references the others

### 11.1 Architecture

- [ ] Create `AGENTS.md` with project decisions and architecture
- [ ] Create `STYLEGUIDE.md` with coding conventions
- [ ] Create `TESTING.md` with testing guidelines
- [ ] Create `docs/architecture.md` with detailed architecture

### 11.2 References

- [ ] Verify all style guides are referenced from AGENTS.md

---

## Phase 12 — Final Verification

### Acceptance Criteria

- [ ] `grep -r ": any" src/` returns zero results
- [ ] `tsc --noEmit` passes with zero errors
- [ ] `npm run lint` passes with zero errors
- [ ] `npm test` passes (all tests)
- [ ] No secrets in tracked files (`git ls-files | xargs grep -l "sk-"` returns empty)
- [ ] `node --version` ≥ 20 and `npm test` passes
- [ ] Every source file (except `*.types.ts` and `index.ts`) has a corresponding `*.test.ts`
- [ ] `grep -r "^export function\|^export async function" src/` returns zero results (no standalone exported functions)

### 12.1 Type Safety

- [ ] Verify zero `any` usage with grep
- [ ] Run `tsc --noEmit` — zero errors
- [ ] Verify no `as` type assertions (except where absolutely necessary)

### 12.2 OOP Compliance

- [ ] Verify zero standalone exported functions: `grep -r "^export function\|^export async function" src/`
- [ ] Verify all runtime modules export classes
- [ ] Verify no module-level singletons or global state

### 12.3 Test-per-File Compliance

- [ ] Verify every `src/<domain>/<name>.ts` has a `tests/<domain>/<name>.test.ts`
- [ ] Exception: `*.types.ts` and `index.ts` files are exempt
- [ ] Run `npm test` — all tests pass

### 12.4 Linting

- [ ] Run `npm run lint` — zero errors
- [ ] Run `npm run format` — consistent formatting

### 12.5 Secrets

- [ ] Verify no secrets in committed files
- [ ] Verify `.env` is in `.gitignore`
- [ ] Verify `.env.example` has only placeholder values

### 12.6 Node Compatibility

- [ ] Verify package.json engines field (>=20.0.0)
- [ ] Test with Node 20

---

## Progress Summary

| Phase             | Total Tasks | Completed | Status  |
| ----------------- | ----------- | --------- | ------- |
| 1 — Setup         | 11          | 11        | ✅ DONE |
| 2 — Types         | 10          | 10        | ✅ DONE |
| 3 — Core          | 8           | 8         | ✅ DONE |
| 4 — LLM Provider  | 8           | 8         | ✅ DONE |
| 5 — Executor      | 5           | 5         | ✅ DONE |
| 6 — Tools         | 4           | 4         | ✅ DONE |
| 7 — Planner       | 6           | 0         | PENDING |
| 8 — Telemetry     | 2           | 0         | PENDING |
| 9 — CLI           | 9           | 0         | PENDING |
| 10 — Cycle        | 2           | 0         | PENDING |
| 11 — Docs         | 4           | 0         | PENDING |
| 12 — Verification | 13          | 0         | PENDING |
| **TOTAL**         | **82**      | **46**    | **56%** |
