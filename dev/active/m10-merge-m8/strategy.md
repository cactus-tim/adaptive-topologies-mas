# m10-merge-m8 — Merge Strategy

## Strategy choice: phased-additive merge (NOT rebase)

PR#10 is already open with review history; rebase would invalidate SHAs in review comments. Strategy: build all m9/m9.1/m9.2 functionality additively on top of feat/m10 (waves 2–8), then a single `git merge origin/feat/m8 --no-commit --no-ff` (wave 9) with file-by-file resolution that keeps our m10 versions of 6 protected files and accepts m8 versions of new files (topologies, conf/, tests/).

Rejected alternatives:
- **rebase feat/m10 onto feat/m8**: would invalidate PR#10 review-comment SHAs; complex multi-commit conflict resolution.
- **`git merge -X ours`**: produces wrong result for add/add (drops m8 new files).
- **`git merge --squash`**: loses m8 commit history granularity on m10.

## Wave-tagging invariant

Before starting ANY wave N>=2, run `git tag -f wave-N-pre HEAD`. This gives per-wave rollback granularity:

```
git reset --hard wave-N-pre  # rollback all changes from wave N
```

Wave-1-pre = m10 HEAD before any work begins. Tag created in Step 1.1.

## Backup

`backup/feat-m10-pre-m8-merge` branch created at wave-1-pre — final fallback if multiple waves need to be unrolled.

## Final merge plan (wave 9)

1. `git merge origin/feat/m8 --no-commit --no-ff`
2. For 6 protected files (`runner.py`, `config.py`, `evaluation/__init__.py`, `evaluation/metrics.py`, `dev/codebase-map.md`, `uv.lock`): `git checkout HEAD -- <path>` to keep m10 versions.
3. Accept default merge for new m8 additions (topologies with HITL kwargs, conf/, tests/).
4. If `src/atm/experiment/_evaluator.py` reappears — `git rm` it.
5. `grep -rn "_evaluator" src/atm/topology/` — strip if found.
6. Alembic single-head check; `alembic merge` if 2 heads.
7. `git commit -m "merge: integrate feat/m8 (m9+m9.2) into feat/m10"`

## Alembic head pre-check (wave 3, step 3.3)

Before creating new migration:
1. `git ls-tree -r origin/feat/m8 -- alembic/versions/`
2. For each m8-only file: `git show origin/feat/m8:alembic/versions/<file>` — inspect for `cognitive_load_proxy` column add.
3. If m8 already adds it → `git checkout origin/feat/m8 -- alembic/versions/<file>` (verbatim).
4. Otherwise → create new with `down_revision = "0001"` and distinct revision id.
5. Final: `uv run alembic heads | wc -l == 1`.

Result of pre-check (to be filled in by step 3.3 agent): _pending_

## Final architectural canon

- `runner.py` = M11-canonical (`compute_quality` + `seed_all` + finally model_version/sandbox_digest) **WITH** M9.2 HITL/RoleRouter layered on top.
- `_evaluator.py` (M6 stub) **must remain deleted**.
- `evaluation/metrics.py` = single file hosting BOTH M11 RQ1-RQ4 functions AND M9.2 `human_sim_cognitive_load_proxy` (per arch/PLAN.md line 612).
- `ExperimentConfig` keeps BOTH `human: HumanCfg | None` AND `evaluation: EvaluationCfg`.
- `Run` model has 4 fields: `human_role`, `cognitive_load_proxy`, `model_version_snapshot`, `sandbox_image_digest`.
