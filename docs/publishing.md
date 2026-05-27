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
- **TestPyPI** (free) — to dry-run releases before promoting to PyPI

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

**Repeat for TestPyPI** at <https://test.pypi.org> with environment name `testpypi`.

**Create the GitHub environments** (so the workflow can reference them):

```bash
# These are simple environment stubs — the workflow file references them
# as `environment: pypi` / `environment: testpypi`. GitHub creates them
# on first run; you can also create them ahead of time via the web UI:
# https://github.com/kkollsga/kglite-docs/settings/environments
```

That's the entire one-time setup.

## 4. Cut a release

When you're ready to ship a version:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `release.yml` workflow fires automatically and:

1. Builds `sdist` + `wheel` (pure-Python, single artifact across OSes).
2. Publishes to **TestPyPI** via trusted publisher.
3. Spins up a fresh venv and runs `pip install kglite-docs==0.1.0` from TestPyPI as a smoke test.
4. Promotes to **PyPI**.
5. Creates a **GitHub Release** with auto-generated notes from commits since the previous tag.

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
| `release.yml` fails at "Publish to TestPyPI" with 403 | Trusted publisher not registered yet | Repeat step 3 |
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
