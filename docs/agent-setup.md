# Claude Code and Codex

Skills have one source under `skills/`. Relative directory symlinks expose the
same source in `.claude/skills/` and `.agents/skills/`. Clone with symlinks enabled.
Both names have standard YAML name/description frontmatter. Run from repository
root after `uv sync --dev`. Agent clients are optional and are not launched by tests.
Native skill discovery was checked structurally, not inside interactive clients.

Use `uv run symbolic-wsp --help`; each command returns JSON. Never use Pi tool
names. Prefer Python APIs for multi-step local experiments. The CLI and APIs do
not implement an autonomous editing agent; Claude Code or Codex supplies hypotheses.

Final data must be on a separate evaluator machine/account as documented in
methodology.md. A prompt prohibition cannot protect data mounted into agent tools.
