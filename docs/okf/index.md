---
okf_version: "0.2"
---

# Fucina knowledge bundle

This directory is the OKF v0.2 knowledge bundle for fucina. It is a dated, source-linked view of architecture, capabilities, implementation maturity, validation, and open work.

# Core concepts

* [System overview](overview.md) - Purpose, scope, platform, and high-level maturity.
* [Roadmap](roadmap.md) - Evidence-ranked implementation priorities.

# Subdirectories

* [Architecture](architecture/) - CUDA runtime and serving-system design.
* [Capabilities](capabilities/) - Supported models and DS4-inspired feature pillars.
* [Implementation](implementation/) - Current status and known gaps at the documented commit.
* [Validation](validation/) - Test gates and measured performance position.

# Bundle conventions

* Concept documents use YAML frontmatter and standard Markdown links.
* `generated` records when this snapshot was produced; no `verified` field is present until a human or independent process confirms a concept.
* Time-sensitive concepts are `draft` and expire through `stale_after`.
* The implementation snapshot is local commit `480f1b85722754d0e692321082890b6103fe56d8`, validated on 2026-07-25; `origin/main` remains at `39a96db`.
