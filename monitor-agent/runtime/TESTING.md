# TESTING.md -- @monitor-agent/runtime

## Framework

- **Jest 29+** with **ts-jest** ESM preset (`ts-jest/presets/default-esm`).
- Test environment: `node`.
- ESM mode enabled via `NODE_OPTIONS='--experimental-vm-modules'` in the `test` script.

## Running Tests

```bash
npm test
```

This runs Jest with ESM support enabled. The command expands to:
```
NODE_OPTIONS='--experimental-vm-modules' jest
```

## Test File Structure

### Location

Tests mirror the `src/` directory structure under `tests/`:

```
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
  setup.test.ts   -- Global test setup
```

### Naming Convention

Test files follow the pattern `tests/<domain>/<name>.test.ts` mirroring `src/<domain>/<name>.ts`.

Examples:
- `src/shared/logger.ts` -> `tests/shared/logger.test.ts`
- `src/executor/circuit-breaker.ts` -> `tests/executor/circuit-breaker.test.ts`
- `src/cli/commands/run.ts` -> `tests/cli/commands/run.test.ts`

### Exemptions

The following file types do **not** require test files:

- `*.types.ts` -- Pure type definitions with no runtime logic.
- `index.ts` -- Barrel re-export files with no logic.

## Test Structure

Every test file uses Jest's `describe`/`it` pattern:

```typescript
import { SomeClass } from "../../src/domain/some-class.js";

describe("SomeClass", () => {
  describe("methodName", () => {
    it("should do something specific", () => {
      // Arrange
      const sut = new SomeClass(/* deps */);

      // Act
      const result = sut.methodName(/* args */);

      // Assert
      expect(result).toEqual(/* expected */);
    });

    it("should handle edge case", () => {
      // ...
    });
  });
});
```

### Conventions

- One `describe` block per class.
- One nested `describe` block per public method.
- `it` descriptions start with `should` and describe the expected behavior.
- Use Arrange-Act-Assert pattern.
- Tests should be independent (no shared mutable state between tests).

## Mocking Patterns

### Constructor Injection Mocking

All dependencies are injected via constructors, making mocking straightforward:

```typescript
describe("CycleRunner", () => {
  it("should run the execution cycle", () => {
    const mockContractLoader = { loadAllContracts: jest.fn() } as unknown as ContractLoader;
    const mockStateManager = { createState: jest.fn() } as unknown as StateManager;
    // ... other mocks

    const runner = new CycleRunner(
      mockContractLoader,
      mockStateManager,
      // ... all other dependencies
    );
  });
});
```

Use `as unknown as ConcreteType` to cast plain mock objects to the expected class type. This is the project's standard mocking pattern -- it avoids creating full class instances while preserving type safety in the test.

### jest.fn() for Spy/Mock Functions

```typescript
const mockMethod = jest.fn().mockReturnValue(expectedValue);
```

### Mocking Module-Level Dependencies

For modules that import external dependencies (e.g., `fs`, `dotenv`), use `jest.mock()`:

```typescript
jest.mock("node:fs", () => ({
  existsSync: jest.fn().mockReturnValue(true),
  readFileSync: jest.fn().mockReturnValue("file content"),
}));
```

### Resetting State Between Tests

Use `beforeEach` to reset mocks and caches:

```typescript
beforeEach(() => {
  jest.clearAllMocks();
  envLoader.resetCache();
});
```

## Module Name Mapping

Jest config maps `.js` imports to extensionless paths for ts-jest compatibility:

```javascript
moduleNameMapper: {
  "^(\\.{1,2}/.*)\\.js$": "$1",
},
```

This allows test files to import from `src/` using the same `.js` extensions used in production code.

## Coverage

### Configuration

```javascript
collectCoverageFrom: ["src/**/*.ts", "!src/types/**"],
coverageDirectory: "coverage",
coverageReporters: ["text", "text-summary"],
```

- Coverage is collected from all `src/**/*.ts` files.
- `src/types/` (barrel exports) is excluded from coverage.
- Reports are displayed in the terminal (`text` and `text-summary`).

### Expectations

- Every non-exempt source file should have a corresponding test file.
- Test files should cover all public methods and key private logic paths.
- Branches (if/else, switch, try/catch) should be exercised where practical.
- Run `npm run typecheck` after writing tests to verify type safety.

## Type Checking Tests

Tests are included in the TypeScript compilation:

```json
"include": ["src/**/*.ts", "tests/**/*.ts"]
```

Run type checking on test files:
```bash
npm run typecheck
```

This uses `tsconfig.test.json` (or falls back to `tsconfig.json`) with `--noEmit`.

## Cross-References

- **Project architecture:** See [AGENTS.md](./AGENTS.md)
- **Coding conventions:** See [STYLEGUIDE.md](./STYLEGUIDE.md)
- **Detailed architecture:** See [docs/architecture.md](./docs/architecture.md)
