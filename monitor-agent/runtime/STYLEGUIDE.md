# STYLEGUIDE.md -- @monitor-agent/runtime

## TypeScript Configuration

- **Strict mode** is enabled (`"strict": true` in tsconfig.json).
- Additional strictness flags: `noUncheckedIndexedAccess`, `noImplicitOverride`, `noPropertyAccessFromIndexSignature`, `noFallthroughCasesInSwitch`, `noUnusedLocals`, `noUnusedParameters`.
- Target: ES2022. Module: NodeNext.
- All imports must use `.js` extensions (ESM with NodeNext resolution).

## OOP Principles

- **Prefer classes over functions.** Every domain module is a class with a clearly defined responsibility. No standalone exported functions except for type-level utilities, constants, and small helper functions that are not part of the public API.
- **Dependency injection via constructors.** All external dependencies are passed through constructor parameters. No module-level singletons, no global state, no `import` of concrete instances at module scope.
- **Interfaces for testability.** Define interfaces (e.g., `EnvLoader` in `llm-config.ts`) to abstract dependencies that need mocking in tests.
- **Readonly by default.** All interface properties and class fields use `readonly`. Mutation is explicit and internal-only (e.g., `MutableState` wrapper in `CycleRunner`).

## Naming Conventions

| Element | Convention | Examples |
|---------|-----------|----------|
| Classes | PascalCase | `CycleRunner`, `ToolExecutor`, `PerceptionBuilder` |
| Interfaces | PascalCase | `AgentState`, `LlmConfig`, `ToolFunction` |
| Type aliases | PascalCase | `LogLevel`, `ActionType`, `HookName` |
| Methods | camelCase | `loadAllContracts()`, `callLlm()`, `validate()` |
| Private fields | camelCase with `private` | `private readonly logger: Logger` |
| Constants | camelCase or UPPER_SNAKE_CASE | `DEFAULT_RETRY_COUNT`, `PROVIDER_DEFAULT_URLS` |
| Contract-facing fields | snake_case (Portuguese) | `proxima_acao`, `nome_ferramenta`, `sucesso` |
| Internal TypeScript fields | camelCase | `proximaAcao`, `nomeFerramenta`, `sucesso` |
| File names | kebab-case | `cycle-runner.ts`, `tool-builder.ts`, `llm-client.ts` |
| Type definition files | `*.types.ts` | `state.types.ts`, `llm.types.ts`, `cycle.types.ts` |

### Why Portuguese field names?

Contract-facing fields (`sucesso`, `dados`, `erro`, `proxima_acao`, `nome_ferramenta`, `criterio_sucesso`) use Portuguese names for backward compatibility with the Python runtime and existing agent YAML contract definitions. Internal TypeScript fields use camelCase (e.g., `proximaAcao`) following standard TypeScript conventions. The `Plan` type maps `proxima_acao` (LLM output) to `proximaAcao` (TypeScript field).

## Type Safety

- **No `any`.** The ESLint rule `@typescript-eslint/no-explicit-any` is set to `error`. Use `unknown` or specific types instead.
- **No `as` assertions.** Avoid type assertions. If an `as` is truly necessary, add a comment explaining why.
- **`import type` for type-only imports.** All type imports must use the `import type` syntax:
  ```typescript
  import type { AgentState } from "./state.types.js";
  ```
  Value imports (classes, functions, constants) use regular `import`.
- **Zod `z.infer` for derived types.** Contract types are derived from Zod schemas using `z.infer<typeof Schema>`, not manually defined.
- **Exhaustive switch checks.** Use `never` type in default branches for exhaustive type narrowing:
  ```typescript
  default: {
    const _exhaustive: never = config.provider;
    throw new Error(`Unsupported provider: ${_exhaustive}`);
  }
  ```

## File Organization

- Each domain is a directory under `src/` with a consistent structure:
  - `<class-name>.ts` -- Class implementation
  - `<domain>.types.ts` -- Type definitions (interfaces, type aliases)
  - `schemas.ts` -- Zod schemas (contracts domain only)
  - `index.ts` -- Barrel export (optional, used by `src/types/`)
- Type files (`*.types.ts`) contain only type definitions, not runtime code.
- Barrel files (`index.ts`) only re-export; no new definitions.

## ESLint Rules

Configured in `eslint.config.mjs`:

| Rule | Setting | Meaning |
|------|---------|---------|
| `@typescript-eslint/no-explicit-any` | `error` | No `any` type |
| `@typescript-eslint/no-unused-vars` | `error` (prefix `_` ignored) | Unused variables are errors; prefix with `_` to suppress |
| `@typescript-eslint/explicit-function-return-type` | `error` | All functions must declare return type |
| `@typescript-eslint/explicit-module-boundary-types` | `error` | Exported functions must have explicit parameter and return types |
| `@typescript-eslint/no-non-null-assertion` | `warn` | `!` assertions are warnings |
| `no-console` | `off` | `console` is allowed (for Logger internals) |

## Formatting

- Prettier is configured for all `src/**/*.ts` files.
- Run `npm run format` to format, `npm run format:check` to verify.

## Secrets and Environment Variables

- **Never commit secrets.** API keys are loaded from `.env` files via `dotenv` and validated with Zod schemas.
- `.env` files must be listed in `.gitignore`.
- Environment variables are accessed through the `EnvLoader` class, not directly via `process.env` outside the shared domain.

## Import Conventions

- Use `.js` extensions in all import paths (ESM NodeNext requirement).
- Use `import type` for type-only imports.
- Group imports: external packages first, then internal modules.
- Alias when naming conflicts arise:
  ```typescript
  import type { SkillParamSchema as SkillParamSchemaType } from "./schemas.js";
  ```

## Comments and Documentation

- JSDoc comments on all public classes, methods, and interfaces.
- JSDoc includes `@param`, `@returns`, `@throws` where applicable.
- `Used by:` and `Acceptance criteria:` annotations in JSDoc for key methods.
- No inline comments explaining obvious code. Comments explain *why*, not *what*.

## Cross-References

- **Project architecture:** See [AGENTS.md](./AGENTS.md)
- **Testing conventions:** See [TESTING.md](./TESTING.md)
- **Detailed architecture:** See [docs/architecture.md](./docs/architecture.md)
