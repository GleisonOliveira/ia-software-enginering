# Project Rules

## Directories to ignore

- Do not read or scan `node_modules` directories
- Ignore any directory listed in `.gitignore` files
- Avoid searching for environment files, lock files, or build artifacts

## Scanning behavior

- Avoid scanning project folders to discover patterns or conventions
- Prefer reading only the relevant lines of a file rather than entire files
- Use targeted searches (grep) instead of broad directory listings

## General rules

- TypeScript and LangGraph projects
- Avoid `any` in type annotations
- Tests should be placed in a `tests` folder within each project, organized by domain
- Tests should prefer using original function types and return values
- Avoid `as` assertions as much as possible
- Use Jest for testing
- Projects must use TypeScript
- Prefer classes over plain functions/objects
- Prefer dependency injection over direct instantiation
- Variables and identifiers must be named in English
- Follow SOLID principles
- Follow TypeScript naming conventions (especially Nest.js style)
- Always use `import type` for type-only imports
- Never store secrets in committable files; always use `.env` files excluded by `.gitignore`
