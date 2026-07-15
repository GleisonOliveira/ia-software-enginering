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

## Phase 1 — Project Setup

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

### 2.1 Shared Types
- [ ] Create `src/shared/shared.types.ts` (TokenUsage, ActionResult, etc.)

### 2.2 LLM Provider Types
- [ ] Create `src/llm/llm.types.ts` (LlmProvider, LlmConfig, StructuredOutputOptions, etc.)

### 2.3 Contract Schemas and Types
- [ ] Create `src/contracts/schemas.ts` (Zod schemas for all contracts)
- [ ] Create `src/contracts/contracts.types.ts` (types derived from Zod schemas)

### 2.4 Core Types
- [ ] Create `src/core/state.types.ts` (AgentState, StateLimits, etc.)
- [ ] Create `src/core/cycle.types.ts` (CycleConfig, HistoryEntry, etc.)

### 2.5 Executor Types
- [ ] Create `src/executor/executor.types.ts` (ExecutionResult, ValidationResult, etc.)

### 2.6 Tools Types
- [ ] Create `src/tools/tools.types.ts` (ToolFunction, ToolDefinition, etc.)

### 2.7 Planner Types
- [ ] Create `src/planner/planner.types.ts` (Plan, Perception, etc.)

### 2.8 Telemetry Types
- [ ] Create `src/telemetry/telemetry.types.ts` (TelemetryEvent, PhaseMarker, etc.)

### 2.9 CLI Types
- [ ] Create `src/cli/cli.types.ts` (RunOptions, ValidateOptions, etc.)

### 2.10 Public Re-exports
- [ ] Create `src/types/index.ts` (public type re-exports)

---

## Phase 3 — Core Domain

### 3.1 Environment
- [ ] Create `src/shared/env.ts` (dotenv loading with Zod validation)
- [ ] Validate `LLM_BASE_URL` env var (provider-specific defaults when empty)

### 3.2 Logger
- [ ] Create `src/shared/logger.ts` (structured logger replacing print statements)

### 3.3 Contract Loader
- [ ] Create `src/contracts/loader.ts` (YAML extraction from .md files)
- [ ] Implement `loadYamlFromMd(filePath)` function
- [ ] Implement `loadAllContracts(agentPath)` function

### 3.4 State Manager
- [ ] Create `src/core/state.ts` (state creation from contracts)
- [ ] Implement `createState(contracts, input, mode?, event?)` function

---

## Phase 4 — LLM Provider Abstraction

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

### 8.1 Telemetry Collector
- [ ] Create `src/telemetry/telemetry.ts`
- [ ] Implement `Telemetry` class
- [ ] Implement event registration, phase timing, token tracking
- [ ] Implement `healthMetrics()`, `performanceData()`, `auditLogs()` methods

---

## Phase 9 — CLI

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

### 11.1 Architecture
- [ ] Create `AGENTS.md` with project decisions and architecture
- [ ] Create `STYLEGUIDE.md` with coding conventions
- [ ] Create `TESTING.md` with testing guidelines
- [ ] Create `docs/architecture.md` with detailed architecture

### 11.2 References
- [ ] Verify all style guides are referenced from AGENTS.md

---

## Phase 12 — Final Verification

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

| Phase | Total Tasks | Completed | Status |
|-------|------------|-----------|--------|
| 1 — Setup | 11 | 11 | ✅ DONE |
| 2 — Types | 10 | 0 | PENDING |
| 3 — Core | 4 | 0 | PENDING |
| 4 — LLM Provider | 5 | 0 | PENDING |
| 5 — Executor | 4 | 0 | PENDING |
| 6 — Tools | 3 | 0 | PENDING |
| 7 — Planner | 3 | 0 | PENDING |
| 8 — Telemetry | 1 | 0 | PENDING |
| 9 — CLI | 2 | 0 | PENDING |
| 10 — Cycle | 1 | 0 | PENDING |
| 11 — Docs | 4 | 0 | PENDING |
| 12 — Verification | 5 | 0 | PENDING |
| **TOTAL** | **53** | **11** | **21%** |
