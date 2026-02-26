# Contributing to GeoScriptHub

Thank you for considering a contribution!  GeoScriptHub is open to bug fixes,
new tools, documentation improvements, and feature enhancements.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)
- [Adding a New Tool](#adding-a-new-tool)

---

## Code of Conduct

Be respectful.  Harassment, discrimination, or dismissive communication will
not be tolerated.

---

## How to Contribute

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally.
3. **Create** a dedicated feature branch (see naming below).
4. **Make** your changes following the code style guidelines.
5. **Test** your changes locally.
6. **Push** and open a Pull Request against `main`.

### Branch Naming

| Prefix | Use for |
|--------|---------|
| `feature/` | New tools or features |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructures with no behaviour change |
| `chore/` | Dependency bumps, CI config, tooling |

Example: `feature/add-osm-export-tool`

---

## Development Setup

### Python tools

```bash
# From the repo root — add shared/ to PYTHONPATH
# Windows:
set PYTHONPATH=.
# macOS / Linux:
export PYTHONPATH=.

# Install a specific tool in editable mode
cd tools/python/<tool-name>
pip install -e ".[dev]"
```

The `[dev]` extra installs `ruff`, `mypy`, and `pytest`.

### TypeScript widgets

```bash
# Requires Node.js 18+ and pnpm
cd tools/typescript/<widget-name>
pnpm install
pnpm dev      # local dev server
pnpm build    # production build
pnpm lint     # ESLint check
```

---

## Code Style

### Python

- **Formatter:** `ruff format` (configure in each tool's `pyproject.toml`)
- **Linter:** `ruff check`
- **Type checker:** `mypy --strict`
- **Docstrings:** Google style on all public classes and methods
- **OOP:** All tools must inherit from `shared.python.GeoTool`
- **Exceptions:** Raise from `shared.python.exceptions`, never use bare `Exception`

### TypeScript

- **Formatter / Linter:** ESLint + Prettier (config in each widget's `package.json`)
- **Types:** `strict: true` in all `tsconfig.json` files — no `any` without a comment justification
- **Classes:** Each widget must be a proper TypeScript class with a clear public API

---

## Submitting a Pull Request

- Keep PRs focused — one feature or fix per PR.
- Write a clear description: what changed and why.
- Reference any related issue: `Closes #42`.
- Ensure CI passes before requesting review.
- Add or update tests if your change affects logic.
- Update the relevant `README.md` if you change a tool's interface.

---

## Reporting Bugs

Open a GitHub Issue using the **Bug Report** template.  Include:

- Which tool is affected.
- Steps to reproduce.
- Expected vs. actual behaviour.
- Python / Node version and OS.
- Any relevant error messages or stack traces.

---

## Adding a New Tool

1. Create `tools/python/<your-tool-name>/` (or `tools/typescript/<name>/`).
2. Follow the directory layout of an existing tool (e.g. `batch-coordinate-transformer`).
3. Python tools **must** subclass `GeoTool` and implement `validate_inputs` + `process`.
4. Include a `pyproject.toml` (Python) or `package.json` + `tsconfig.json` (TypeScript).
5. Write a `README.md` using the same structure as existing tools.
6. Add the tool to the **Tool Index** table in the root `README.md`.
7. Open a PR — the CI will validate your code automatically.
