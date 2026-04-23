# Lizard — SIMULA Agent

> Terminal-first, multi-agent **life situation simulator** grounded in the first
> four steps of the Buddha's Noble Eightfold Path:
> **Right Intention → Right Thought → Right Speech → Right Action**.

Lizard spins up a structured group of **5 to 28** agents that deliberate a
life situation you feed it, step by step, and produces a **Dhamma report**
(JSON + Markdown) at the end. It is explicitly designed to be:

- **Terminal-only** — no web server, no GUI. Runs in a plain shell.
- **Local-first** — persistence in a SQLite database, filesystem, and
  in-memory structures. LLM reasoning is done by a **local Ollama** model
  (`gemma3` by default); when Ollama is unavailable, Lizard degrades
  deterministically so pipelines never break.
- **Production-grade Python** — PEP 8, typed, ruff-clean, pytest-covered,
  with a `Makefile` for build/lint/test/run/validate.

## Theoretical lineage

Lizard's reasoning layer is a cooperative ensemble of classical AI and
physics techniques:

| Module              | Concept                                                 |
| ------------------- | ------------------------------------------------------- |
| `lizard.search`     | BFS, DFS, UCS, A\*, beam search (Russell & Norvig)      |
| `lizard.game_theory`| Payoff matrices, Nash-like best response, cooperation   |
| `lizard.thermodynamics` | Shannon entropy, softmax-Boltzmann action selection |
| `lizard.world`      | Catalogue of "world technologies" (tools agents use)    |
| `lizard.llm`        | Local Ollama client (`gemma3`) with deterministic fallback |
| `lizard.simula`     | Agent orchestrator, step runner, Dhamma phases          |

## Install

```bash
git clone <this-repo> lizard
cd lizard
make install        # editable install + dev extras
cp .env.example .env
```

Optional (recommended): install and run [Ollama](https://ollama.com/) and pull
a local model:

```bash
ollama pull gemma3
ollama serve &
```

If Ollama is not running, Lizard uses a deterministic reasoning stub so every
command still works offline.

## CLI

```bash
lizard --interactive
lizard --run-simula --prompt "Two people getting arrested for mismanagement of records"
lizard --view-stats
lizard --reports
lizard --generate-report
lizard --load-system
```

| Command              | Description                                                                |
| -------------------- | -------------------------------------------------------------------------- |
| `--interactive`      | Boot, ask for a prompt, then walk the simulation step-by-step on `Enter`.  |
| `--run-simula`       | Run a full simulation headlessly for the supplied `--prompt`.              |
| `--view-stats`       | Print aggregate run stats from the SQLite store.                           |
| `--reports`          | List past reports and pretty-print a selected one.                         |
| `--generate-report`  | Prompted: type a situation and dump the agent log to JSON + Markdown.      |
| `--load-system`      | Re-ingest all outputs/logs and rebuild the SIMULA config (relearn).        |

## Data layout

```
~/.lizard/
├── lizard.db           # SQLite (runs, agents, steps, reports, configs)
├── reports/
│   ├── 2026-04-22T16-30-00Z.json
│   └── 2026-04-22T16-30-00Z.md
└── configs/
    └── simula-2026-04-22T16-30-00Z.json
```

Override the root with `LIZARD_HOME` in `.env`.

## Development

```bash
make lint       # ruff + mypy
make test       # pytest
make validate   # lint + test
make run        # lizard --interactive
make build      # sdist + wheel
```

## License

MIT — see `LICENSE`.
