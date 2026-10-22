# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-22

### Breaking

- Packs where a whole module has no temperature data are now rejected with exit code 2 instead of being analysed (see Added); files that 0.2.0 accepted can therefore fail until the sensor wiring or the CSV is fixed (95d5763).
- Analysis configs that set `rest.m6_slope_weight` or `rest.m6_max_slope_contribution` are rejected as unknown keys; delete those lines.
- 2P16S packs analysed with 0.2.0 or earlier should be re-run: their module 2 and 4 temperatures were missing, so the earlier verdicts are not trustworthy.

### Fixed

- 2P16S packs: temperature columns are now read using the tester's per-module-pair layout. Modules 2 and 4 previously loaded with no temperatures, which silently distorted every module's z-scores (a5ba71e).
- Tester filenames with year-first dates (`_D20261002_`) now give the correct test date instead of "unknown" (13d4b14).
- `spread`: the MAD floor is now one voltage quantisation step over the rest duration, so a single 1 mV step over the rest no longer scores as an outlier on its own, while a genuine drift of a few millivolts still scores HIGH (e02b055).
- PDF export: Plotly is pinned below 7, which dropped support for the bundled Kaleido 0.2 renderer, so fresh installs and release builds still produce PDF reports (5d4f47d).

### Changed

- The `rank` method (M6) is report-only: its chart and statistics stay, but it has no z-score and never enters the composite or the `fleet_divergence` cluster, which is now `spread` alone; reports and README describe seven scored methods. Composite values of healthy cells move slightly; no cell or module verdicts changed on the reference packs (6eb7924).

### Added

- A pack with a module that has no temperature data at all is now rejected with exit code 2 ("Test invalidated") instead of producing a silently wrong report (95d5763).
- The golden regression test selects its float tolerance from the platform that generated the golden file (d4e4b3b).
- This changelog.

## [0.2.0] - 2026-07-07

Previous release; see the GitHub release notes.

## [0.1.0] - 2026-07-01

Previous release; see the GitHub release notes.

[Unreleased]: https://github.com/Ozzstein/stress-screen/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Ozzstein/stress-screen/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ozzstein/stress-screen/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ozzstein/stress-screen/releases/tag/v0.1.0
