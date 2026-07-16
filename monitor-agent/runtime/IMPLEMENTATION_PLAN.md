# Implementation Plan — Runtime TypeScript

> **Status:** PENDING
> **Created:** 2026-07-15
> **Target:** `monitor-agent/runtime/`
> **Last Updated:** 2026-07-15

## Overview

Conversion of the Python AI agent execution runtime to TypeScript with:

- Domain-based architecture
- Zod schema validation
- Dependency injection via classes
- Strong typing (no `any`)
- Jest unit tests
- ESLint + Prettier
- Node 20+ compatibility

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

### Function/method/type comments

- Every exported function, method, type alias, and interface must have a JSDoc comment explaining:
  - **What** it does/is (brief description)
  - **Why** this approach was chosen (design rationale when non-obvious)
  - **Where** it is used and in what context (e.g., "Used by: cycle runner, planner")
  - `@param` tags for all parameters (functions/methods)
  - `@returns` description (functions/methods)
  - `@typeParam` for generic type parameters
- For type aliases and interfaces, document each field with a brief inline comment if the name alone is not self-explanatory.

### Inline comments

- Use inline comments (`//`) to explain non-obvious logic, workarounds, or domain-specific rules.
- Do **not** restate what the code does — focus on intent and reasoning.
- Comment complex business rules (circuit breaker logic, stagnation detection, token limits) to connect the code back to the contracts.

### Why comments are mandatory

- The Python runtime used Portuguese docstrings extensively for this purpose. The TypeScript version must maintain the same level of clarity.
- The agent contracts (YAML `.md` files) define _what_ the system should do; comments in code explain _how_ and _why_ it is implemented that way.
- Future maintainers and AI agents reading the code need context that type signatures alone do not convey.

### Example

```typescript
/**
 * Validates the LLM response against available tools before execution.
 *
 * Why: The LLM may return invalid tool names or malformed arguments.
 * This circuit breaker prevents runtime errors and enables auto-correction.
 *
 * @param plan - The structured plan returned by the LLM
 * @param availableTools - Set of tool names currently registered
 * @returns List of validation problems (empty = valid)
 */
function validateLlmResponse(plan: Plan, availableTools: Set<string>): string[] {
  // Check if proxima_acao is one of the valid action types defined in contracts
  const validActions = new Set(["CHAMAR_FERRAMENTA", "FINALIZAR", "PERGUNTAR_USUARIO"]);
  // ...
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

- [x] `loadYamlFromMd()` extracts YAML from a test `.md` file correctly
- [x] `loadAllContracts()` loads all 9 contracts from a valid agent_path
- [x] `createState()` returns an `AgentState` with all required fields populated
- [x] `loadEnv()` validates variables with Zod and returns defaults for `LLM_BASE_URL`
- [x] Unit tests with mocked provider pass

### 3.1 Environment

- [x] Create `src/shared/env.ts` (dotenv loading with Zod validation)
- [x] Validate `LLM_BASE_URL` env var (provider-specific defaults when empty)

### 3.2 Logger

- [x] Create `src/shared/logger.ts` (structured logger replacing print statements)

### 3.3 Contract Loader

- [x] Create `src/contracts/loader.ts` (YAML extraction from .md files)
- [x] Implement `loadYamlFromMd(filePath)` function
- [x] Implement `loadAllContracts(agentPath)` function

### 3.4 State Manager

- [x] Create `src/core/state.ts` (state creation from contracts)
- [x] Implement `createState(contracts, input, mode?, event?)` function

---

## Phase 4 — LLM Provider Abstraction

### Acceptance Criteria

- [ ] `LlmClient.callLlm()` returns text and `TokenUsage` with configurable provider via env
- [ ] `resolveProvider()` selects the correct provider based on `LLM_PROVIDER`
- [ ] `generateStructuredOutput()` validates LLM output against a Zod schema
- [ ] Unit tests with mocked provider pass

### 4.1 LLM Client (Provider-Agnostic)

- [ ] Create `src/llm/llm-client.ts`
- [ ] Implement `LlmClient` class wrapping Vercel AI SDK `generateText`
- [ ] Implement provider auto-detection from env vars
- [ ] Implement `callLlm(options)` with structured output support via `Output.object()`
- [ ] Implement token usage extraction from AI SDK response

### 4.2 LLM Provider Config

- [ ] Create `src/llm/llm-config.ts`
- [ ] Implement `resolveProvider()` — reads `LLM_PROVIDER` env, selects provider
- [ ] Support provider-specific env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, etc.
- [ ] Implement `LLM_BASE_URL` env var — custom endpoint for OpenRouter, local models, proxies
- [ ] Implement model selection from `LLM_MODEL` env var
- [ ] Implement token limits from `LLM_MAX_TOKENS` env var
- [ ] Implement structured output schema conversion (Zod → provider format)

### 4.3 Structured Output Helpers

- [ ] Create `src/llm/structured-output.ts`
- [ ] Implement `generateStructuredOutput<T>(schema, prompt, systemPrompt)` using AI SDK `Output.object()`
- [ ] Implement validation fallback for providers with lower structured output reliability
- [ ] Implement retry logic with Zod validation error feedback

---

## Phase 5 — Executor Domain

### Acceptance Criteria

- [ ] `validatePayload()` returns error list for invalid args (empty list for valid args)
- [ ] `validateLlmResponse()` detects invalid plans and applies auto-correction
- [ ] `executeTool()` executes a tool and returns `ActionResult`
- [ ] `evaluate()` returns correct `objetivo_alcancado` and `qualidade`

### 5.1 Payload Validator

- [ ] Create `src/executor/payload-validator.ts`
- [ ] Implement `validatePayload(toolName, args, contracts)` using Zod

### 5.2 Circuit Breaker

- [ ] Create `src/executor/circuit-breaker.ts`
- [ ] Implement `validateLlmResponse(plan, availableTools)` function
- [ ] Implement auto-correction and fallback logic

### 5.3 Tool Executor

- [ ] Create `src/executor/executor.ts`
- [ ] Implement `executeTool(toolName, args, tools, contracts)` function
- [ ] Implement retry logic from contracts

### 5.4 Evaluator

- [ ] Create `src/executor/evaluator.ts`
- [ ] Implement `evaluate(plan, actionResult, contracts)` function
- [ ] Implement output validation against schema

---

## Phase 6 — Tools Domain

### Acceptance Criteria

- [ ] `buildTool()` returns an executable function from a skill definition
- [ ] `ToolRegistry` allows lookup by tool name
- [ ] `executeHook()` fires configured hooks

### 6.1 Tool Builder

- [ ] Create `src/tools/tool-builder.ts`
- [ ] Implement `buildTool(skill)` function
- [ ] Implement LLM-based tool execution with mock fallback

### 6.2 Tool Registry

- [ ] Create `src/tools/tool-registry.ts`
- [ ] Implement `ToolRegistry` class for tool lookup

### 6.3 Hooks

- [ ] Create `src/tools/hooks.ts`
- [ ] Implement `executeHook(name, hookContract, params)` function

---

## Phase 7 — Planner Domain

### Acceptance Criteria

- [ ] `buildPerception()` returns a string with entry, mode, and progress info
- [ ] `buildSystemPrompt()` includes instructions for all modes (interactive, goal_oriented, autonomous)
- [ ] `callLlm()` returns a valid `Plan` with correct `proxima_acao` type
- [ ] `mockPlanner()` works without API key

### 7.1 Perception Builder

- [ ] Create `src/planner/perception.ts`
- [ ] Implement `buildPerception(state)` function

### 7.2 Prompt Builder

- [ ] Create `src/planner/prompt-builder.ts`
- [ ] Implement `buildSystemPrompt(contracts)` function
- [ ] Implement mode-specific instructions (interactive, goal_oriented, autonomous)

### 7.3 Planner

- [ ] Create `src/planner/planner.ts`
- [ ] Implement `callLlm(perception, contracts, history)` using `LlmClient` with structured output
- [ ] Implement `mockPlanner(perception, contracts, history)` fallback

---

## Phase 8 — Telemetry Domain

### Acceptance Criteria

- [ ] `Telemetry` registers events with timestamp and trace_id
- [ ] `healthMetrics()` returns success rate and counters
- [ ] `performanceData()` returns phase timing with avg/max/total

### 8.1 Telemetry Collector

- [ ] Create `src/telemetry/telemetry.ts`
- [ ] Implement `Telemetry` class
- [ ] Implement event registration, phase timing, token tracking
- [ ] Implement `healthMetrics()`, `performanceData()`, `auditLogs()` methods

---

## Phase 9 — CLI

### Acceptance Criteria

- [ ] `run` executes an agent with `--agente` and `--entrada`
- [ ] `validate` validates contracts and returns ok/failure
- [ ] `trace` displays the last trace.json
- [ ] `analyze` generates `analise-agente.md`
- [ ] `replay` re-executes with the same input

### 9.1 Entry Point

- [ ] Create `src/cli/index.ts` with Commander.js

### 9.2 Commands

- [ ] Create `src/cli/commands/run.ts` — run agent
- [ ] Create `src/cli/commands/validate.ts` — validate contracts
- [ ] Create `src/cli/commands/trace.ts` — show last trace
- [ ] Create `src/cli/commands/analyze.ts` — analyze trace
- [ ] Create `src/cli/commands/replay.ts` — replay last execution

---

## Phase 10 — Core Cycle

### Acceptance Criteria

- [ ] `run()` executes the full perceive→plan→act→evaluate cycle
- [ ] Time, token, and stagnation limits interrupt the cycle correctly
- [ ] `replay()` re-executes with input from the previous trace
- [ ] `trace.json` is saved with all telemetry data
- [ ] KPI panel is displayed after each step

### 10.1 Main Cycle

- [ ] Create `src/core/cycle.ts`
- [ ] Implement `run(agentPath, input, mode?, event?, output?)` function
- [ ] Implement `replay(agentPath)` function
- [ ] Implement `showTrace()` function
- [ ] Implement KPI panel display
- [ ] Implement stagnation detection
- [ ] Implement time/token limit checks
- [ ] Implement mandatory tool enforcement
- [ ] Implement sensitive action confirmation
- [ ] Implement trace file saving

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

### 12.1 Type Safety

- [ ] Verify zero `any` usage with grep
- [ ] Run `tsc --noEmit` — zero errors
- [ ] Verify no `as` type assertions (except where absolutely necessary)

### 12.2 Linting

- [ ] Run `npm run lint` — zero errors
- [ ] Run `npm run format` — consistent formatting

### 12.3 Testing

- [ ] Run `npm test` — all tests pass
- [ ] Verify external dependencies are mocked in all tests

### 12.4 Secrets

- [ ] Verify no secrets in committed files
- [ ] Verify `.env` is in `.gitignore`
- [ ] Verify `.env.example` has only placeholder values

### 12.5 Node Compatibility

- [ ] Verify package.json engines field (>=20.0.0)
- [ ] Test with Node 20

---

## Progress Summary

| Phase             | Total Tasks | Completed | Status  |
| ----------------- | ----------- | --------- | ------- |
| 1 — Setup         | 11          | 11        | ✅ DONE |
| 2 — Types         | 10          | 10        | ✅ DONE |
| 3 — Core          | 4           | 4         | ✅ DONE |
| 4 — LLM Provider  | 5           | 0         | PENDING |
| 5 — Executor      | 4           | 0         | PENDING |
| 6 — Tools         | 3           | 0         | PENDING |
| 7 — Planner       | 3           | 0         | PENDING |
| 8 — Telemetry     | 1           | 0         | PENDING |
| 9 — CLI           | 2           | 0         | PENDING |
| 10 — Cycle        | 1           | 0         | PENDING |
| 11 — Docs         | 4           | 0         | PENDING |
| 12 — Verification | 5           | 0         | PENDING |
| **TOTAL**         | **53**      | **25**    | **47%** |
