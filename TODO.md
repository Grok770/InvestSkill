# TODO — InvestSkill

The curated top five. Everything else — with rationale, effort estimates, and release mapping — lives in the bilingual roadmap: [doc/IMPROVEMENT-ROADMAP.md](doc/IMPROVEMENT-ROADMAP.md) · [繁體中文](doc/IMPROVEMENT-ROADMAP-zh-TW.md). Progress against the roadmap's ten headline items is tracked in its §0 table.

The previous long-form backlog (last updated 2026-02-24) is archived at [doc/archive/TODO-2026-02.md](doc/archive/TODO-2026-02.md).

## Top 5

1. **`etf-analysis`** — ETF / index-fund due diligence: expense ratio vs. category, tracking difference, holdings overlap with the user's positions, structure warnings. The single largest blind spot for a US-retail audience. → roadmap §3.1 · target 1.12.0
2. **`earnings-preview`** — the *before*-earnings skill: consensus and whisper gap, 8-quarter beat/move history, implied vs. realized move, three-scenario grid. `earnings-call-analysis` is post-call; `catalyst-calendar` only lists the date. → roadmap §3.1
3. **`full-report --depth comprehensive` runs every framework** — add the six missing since v1.8 (`bear-case` above all), swap the alias modules for their targets, add `--skip <skill>`. → roadmap §4.3
4. **`tax-lens` + Non-US Investor Guide** — US tax mechanics for a position, plus a `--non-us` module (W-8BEN, withholding, estate-tax exposure, UCITS alternatives) and a bilingual site guide. Essential for the zh-TW audience. → roadmap §3.1, §5.4
5. **`--lang zh-TW` on every skill + a machine-readable JSON footer** — the remaining two items of the skill contract (§4.1); the footer is what lets `eval-skills.js` and `result-validator` parse results without regex on box-drawing characters.

## Recently shipped from the roadmap

- Stale counts fixed and historical docs archived (§7)
- Skill contract enforced on every analysis skill — `Data & Sources` header, Data Verification gate, Thesis Invalidation — with `scripts/check-skill-contract.js` in `npm test` (§4.1, §6.3)
- The three redirect skills reclassified as aliases; honest count of 24 frameworks (§4.2)
- `thesis-tracker` (§3.1)
- `scripts/sync-prompts.js`, `scripts/new-skill.js`, `scripts/eval-skills.js`, `scripts/lib/signal-block.js` (§6.1, §6.2, §6.4, §6.11)

## Deliberately not planned

Live-data / broker APIs, crypto and forex, ML price prediction, and changes to the signal block — see roadmap §9 for why.
