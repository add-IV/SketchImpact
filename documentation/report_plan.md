# Report Writing Plan

Working plan for the final report. Kept separate from `plan.md` (the original
project proposal) so that document stays untouched as a reference.

## Outline → source material

Suggested section order and where the content already lives. Target total length
is ~10-12 pages (excluding title page, references, figures); page estimates below
assume standard report formatting (~500 words/page).

1. **Introduction / Problem Statement** — `plan.md` (problem statement, game description) — **~1 page**
2. **Related Work** — `papers.md` (CNN collision papers + classical background) — **~1.5 pages**
3. **Game / Baseline Implementation** — `CLAUDE.md` architecture section (scene tree,
   quadtree broad-phase, pixel-perfect narrow-phase, elastic resolution) — **~1.5 pages**
4. **Dataset Generation** — `dataset/design.md` (Dataset section), `dataset/generate_dataset.py`,
   `dataset/pixel_collision.py` (ground-truth label logic) — **~1.5 pages**
5. **Model Design & Training** — `dataset/design.md` (Input Representation, Augmentation,
   Architecture, Training tables) — largely copy-ready — **~1.5 pages**
6. **Integration into the Game** — `collision_model.gd`, `inference/inference_server.py`
   (stdin/stdout protocol, fail-open behavior), HUD stats (`stats_hud.gd`) — **~1.5 pages**
7. **Evaluation** — see checklist below; this is the section with the biggest gap — **~2.5 pages**
   (biggest section: figures + tables + scenario comparison + baseline discussion)
8. **Conclusion / Future Work** — **~0.5-1 page**

## Checklist

### Data to generate before writing the Evaluation section
- [ ] Run `dataset/eval.py` and save the console output (recall/precision/skip-rate
      table + confusion matrix at the 99.9%-recall operating point) into the repo
      (e.g. `dataset/eval_results.txt`) instead of only printing to stdout
- [ ] Plot the recall-vs-fraction-skipped curve from `eval.py`'s `_curve()` output and
      save it as a figure for the report
- [ ] Decide on and fill in `scenarios/scenario_0.json` (currently empty `sprites: []`),
      or drop it from the scenario comparison
- [ ] For each populated scenario, run the game with the CNN filter **on** and **off**
      (toggle via `enabled` on `collision_model.gd` / eval_mode key) and record the HUD
      numbers (Quadtree pairs, AABB hits, Pixel checks, Model skips, FP/FN, skip rate) —
      screenshot or copy the values, this is the core "vs. baseline" evidence
- [ ] Note the honest baseline comparison: a precomputed O(1) lookup table would beat
      the CNN on every engineering axis (exact, no training, native Godot integration);
      the CNN is included because it's the required academic contribution, not because
      it's the better engineering choice — state this explicitly rather than overselling
      the CNN's practical value

### Writing checklist
- [ ] Introduction: problem statement + motivation (why narrow-phase is expensive for
      freeform sprites)
- [ ] Related work: summarize the 5 recommended papers from `papers.md`, tie each back
      to how it informed (or didn't inform) this project's approach
- [ ] Game/baseline: describe quadtree broad-phase + pixel-perfect narrow-phase,
      include the architecture diagram/scene tree
- [ ] Dataset: describe generation process, label ground truth (FFT cross-correlation),
      train/val split by pair key, class balance (~25% positive)
- [ ] Model: input representation (6×32×32 channels), architecture diagram, augmentation
      (D4 dihedral), training hyperparameters
- [ ] Integration: describe the stdin/stdout IPC design and fail-open safety behavior
      (why it matters for a game — never blocks gameplay if the model breaks)
- [ ] Evaluation: recall/skip-rate curve, confusion matrix, scenario benchmark
      comparison (on vs. off), honest lookup-table baseline discussion
- [ ] Conclusion: what worked, what the CNN approach cost vs. gained, future work
      (e.g. replacing narrow-phase entirely, larger/more diverse sprite dataset)

### Final pass
- [ ] Check all group members are credited consistently with `plan.md`
- [ ] Verify all figures/tables referenced in text actually exist in the repo
- [ ] Proofread; confirm terminology matches code (e.g. "narrow-phase", "AABB",
      "quadtree") consistently throughout
