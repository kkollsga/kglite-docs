# Publishing to GitHub + PyPI — a walkthrough

A one-time setup for shipping `kglite-docs` to GitHub and PyPI using trusted publishing (OIDC — no long-lived tokens on disk).

## 1. Prereqs

```bash
brew install gh   # GitHub CLI; or follow your distro's instructions
gh auth login     # one-time browser login
```

You'll also need accounts on:

- **GitHub** (free) — for the repo + Actions
- **PyPI** (free) — to host the package

## 2. Initialise + push the repo

From the project root:

```bash
git init -b main
git add .
git commit -m "Initial commit: kglite-docs v0.1"

# Create a public repo under your account and push in one command
gh repo create kglite-docs \
    --public \
    --description "Agent-first PDF knowledge base on kglite + bge-m3" \
    --source=. \
    --push
```

Check the result:

```bash
gh repo view --web
```

CI starts running automatically once `.github/workflows/ci.yml` lands on `main`. Watch it:

```bash
gh run watch
```

## 3. Set up PyPI trusted publishing (one-time, manual)

Trusted publishing lets GitHub Actions push to PyPI without an API token — GitHub's OIDC token is verified by PyPI directly.

**Steps on PyPI:**

1. Log in at <https://pypi.org>.
2. Go to your account → **Publishing** → **Pending publishers**.
3. Click **Add a new pending publisher**.
4. Fill in:
   - **PyPI Project Name**: `kglite-docs`
   - **Owner**: `kkollsga` (or your GitHub username)
   - **Repository name**: `kglite-docs`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
5. Save.

**Create the GitHub environment** (so the workflow can reference it):

The `release.yml` workflow uses `environment: pypi`. GitHub creates it on first run; you can also create it ahead of time at <https://github.com/kkollsga/kglite-docs/settings/environments>.

That's the entire one-time setup.

## 4. Cut a release

No tag needed — version bumps are driven by `pyproject.toml`. To cut a release:

1. **Bump the version**:
   ```toml
   # pyproject.toml
   version = "0.0.2"
   ```
2. **Add a changelog entry** in `CHANGELOG.md` under `## [0.0.2] — YYYY-MM-DD`. The release workflow extracts this block and uses it as the GitHub Release body.
3. **Commit + push** to `main`.

The `release.yml` workflow then:

1. Waits for `ci.yml` to pass (won't even start if CI fails).
2. Reads the `version` from `pyproject.toml` and queries `https://pypi.org/pypi/kglite-docs/<version>/json`. HTTP 404 → publish; anything else → skip cleanly.
3. Builds `sdist` + `wheel` (pure-Python, single artifact across OSes).
4. Publishes to **PyPI** via trusted publisher (with `skip-existing: true` as a belt-and-suspenders).
5. Creates a **GitHub Release** at `v<version>` with the CHANGELOG entry as the body. Falls back to auto-generated commit-based notes if the version isn't in `CHANGELOG.md`.

The CI matrix (`ci.yml` on every push) already exercises the package on macOS + Linux × Python 3.10–3.13, so the build → publish path doesn't need a TestPyPI dry-run on top.

Watch it:

```bash
gh run watch
```

## 5. Iterate

For every change going forward:

- Open a PR → CI runs `ruff`, `mypy`, and `pytest`.
- Merge → CI runs again on `main`.
- When ready, push a new tag (`v0.1.1`, `v0.2.0`, etc.) — the release pipeline takes over.

`hatch-vcs` derives `__version__` from the tag automatically; no manual version bumps in source.

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `release.yml` fails at "Publish to PyPI" with 403 | Trusted publisher not registered yet, or environment name mismatch | Repeat step 3; check the env name is `pypi` |
| `Module 'kglite_docs' has no attribute '__version__'` after install | sdist built without `.git` available | Make sure the workflow checks out with `fetch-depth: 0` (the shipped workflow does) |
| `pip install kglite-docs` fails with kglite resolver error | kglite has a stricter Python version constraint | Check the kglite version pin in `pyproject.toml` and tighten if needed |
| CI test failures only on Windows | We don't target Windows in v0.1 | `ci.yml` matrix is mac + linux; revisit if Windows users show up |

## 7. Optional: GitHub Pages docs

`docs/` already holds `getting-started.md`, `architecture.md`, `workflows.md`, `perf.md`, `contributing.md`, and `publishing.md`. To publish them with a static site generator:

```bash
pip install mkdocs-material
mkdocs serve   # local preview at http://localhost:8000
```

Add a `mkdocs.yml` + `gh-pages` workflow when you want the rendered site at `https://kkollsga.github.io/kglite-docs/`. v0.1 ships the raw markdown; the mkdocs setup is a nice-to-have.
