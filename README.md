# qrcheck

`qrcheck` is an offline conformance harness that tests a QR code **generator**: it drives a target you declare in configuration, decodes every symbol that target produces with two independent reference decoders, and reports payload fidelity, symbol conformance, rendering contract, damage resilience, invalid-input handling, determinism and performance against a golden encoder used only as an oracle.

Each case becomes one request built from your configuration, the target returns one artefact, and the framework reads that artefact three ways. `zxing-cpp` 2.2.0 is the primary decoder and the authority for every assertion; `opencv-python-headless` 4.10.0.84 is a second opinion, so two decoders that report different payloads are themselves a finding; and the module grid is sampled with the format information decoded, which recovers the version and mask that neither pinned decoder reports, and yields a second, independent reading of the error-correction level. `segno` 1.6.6 supplies the expectations and nothing else — the oracle, never the subject, always called as `segno.make_qr` with `boost_error=False`, so neither a capacity boost nor a Micro QR symbol can change the expected level or dimension.

## What it is, and what it is not

It is:

- **A conformance harness for generators**, not for scanners. Model 2 QR only, driven by one configuration file that maps the framework's request fields onto your target's inputs and outputs.
- **Offline and single-process.** No host other than `127.0.0.1`/`localhost`, no paid service, no cloud credential, no telemetry, no daemon.
- **Deterministic.** One seed derivation function, sorted iteration in every report and evidence listing, and no wall-clock, locale or hash-order dependence in any assertion — so a run is reproducible from what its report records.
- **Report-producing.** Every run writes `summary.json`, `junit.xml` and a self-contained `index.html`, plus a directory of evidence for every failed, errored or skipped case.
- **Self-verifying.** It ships a conforming `reference` target and a deliberately `defective` target, so the suite is provably not vacuous.
- **Size-gated.** It measures its own tracked physical line count against a 60,000-line ceiling and exits non-zero at it.

It is not:

- **A scanner.** It never reads a photograph, a camera feed or a printed page.
- **A camera or print workflow.** No optics, no hardware, no calibration, no printer.
- **A service.** No HTTP server, no daemon, no account, no database, and no UI beyond one static HTML file.
- **A retry harness.** No retry, re-run or flaky-quarantine path exists anywhere: a result that changes between runs is a framework defect to fix, not to suppress.

## Requirements

- **Interpreter: CPython 3.11.** A newer 3.x may run but is unsupported, and `qrcheck doctor` reports the interpreter as degraded; anything below 3.11 is blocked at start-up because the configuration parser is the standard library's `tomllib`.
- **Platform: Linux x86_64, or macOS on arm64 or x86_64.** Windows is out of scope: no Windows code path exists and no Windows-only dependency is installed.
- **No hardware.** No camera, printer, browser, GPU or Docker daemon is needed or used.
- **No network beyond loopback.** An `http` target must be `127.0.0.1`, `localhost` or `::1`; configuration validation rejects any other host with exit 2, and no other outbound client exists in the dependency set.
- **One optional external tool.** `RN-SVG` needs an SVG rasterizer, discovered at run time in this order: an importable `cairosvg` module, then `rsvg-convert`, `resvg`, `magick` and `convert` on `PATH`; `[render] svg_rasterizer` forces one. With none present, `RN-SVG` reports `skipped` with reason `missing_dependency`, never a pass. The rasterizer is not a Python dependency.
- **One known platform gap.** `zxing-cpp` 2.2.0 publishes no cp311 wheel for Linux aarch64, so on that platform `qrcheck doctor` and `qrcheck run` exit 2 naming the missing distribution. That is a blocking failure by design, not a degradation: the primary decoder is the authority for every check, so its absence cannot be turned into a skip.

## Install

```bash
python3.11 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements.lock
.venv/bin/pip install --no-deps -e .
.venv/bin/qrcheck --help
```

`pip install -e .` puts the `qrcheck` console script in the environment; `python -m qrcheck` is equivalent. Contributors install the development lock instead — it is a superset that starts from the runtime pins, so it cannot disagree with the runtime lock about a shared version:

```bash
.venv/bin/pip install --require-hashes -r requirements-dev.lock
```

Both locks are generated, never hand-edited, with `pip-compile --generate-hashes` under **pip-tools 7.5.3 and pip 24.0** — pip-tools 7.5.3 imports `pip._internal` and breaks on newer pip. The seven direct runtime pins (`Pillow==10.4.0`, `click==8.1.8`, `httpx==0.27.2`, `numpy==1.26.4`, `opencv-python-headless==4.10.0.84`, `segno==1.6.6`, `zxing-cpp==2.2.0`) are declared in `requirements.in` and repeated in [`pyproject.toml`](pyproject.toml), so an editable install cannot drift; [`requirements.lock`](requirements.lock) and [`requirements-dev.lock`](requirements-dev.lock) remain the sole authority for every version, transitive ones included. Regenerating a lock after a pin changes is [`docs/dependencies.md`](docs/dependencies.md)'s job.

## Quickstart

Run these from the repository root, where [`qrcheck.toml`](qrcheck.toml) is the default configuration.

```bash
qrcheck doctor --adapter reference       # verify target and toolchain before testing
qrcheck run --adapter reference          # full run against the conforming target
qrcheck run --adapter defective          # exits 1: exactly the nine checks listed below fail
qrcheck list                             # the 21 registered checks
qrcheck corpus build --seed 20240101     # regenerate corpus/default.jsonl from the seed
qrcheck loc --limit 60000                # tracked physical lines against the ceiling
```

What you should see:

- **`doctor`** prints one line per prerequisite with a status of `ok`, `missing` or `degraded`, separates a blocking prerequisite from a merely degrading one, and exits 0 only when every blocking one passes.
- **The `reference` run** finishes with `failed: 0`, `error: 0` and `skipped: 0`, and every one of the 21 checks has at least one passing result. Open `reports/index.html` for the verdict and the evidence, and `reports/summary.json` for the machine reading.
- **The `defective` run** exits 1 and fails exactly nine checks — `RT-ROUNDTRIP`, `MD-CHARSET`, `SY-VERSION-MIN`, `SY-EC-MATCH`, `SY-MATRIX-EQUIV`, `RN-QUIET-ZONE`, `RN-MODULE-SIZE`, `DT-DETERMINISTIC` and `IV-OVERCAPACITY` — while the remaining twelve pass in the same run. CI asserts that set by equality, so a tenth failure and a missing one are both defects.
- **`list`** prints every check with its family, required capabilities and optional dependencies, sorted by check ID.
- **`corpus build`** rewrites `corpus/default.jsonl` from the seed and reports whether the content changed; `--force` is required to overwrite a file whose content differs.
- **`loc`** prints a per-file table and a total, and exits 0 only while the total is strictly under the limit.

Once the default run is green, narrow it instead of editing the file:

```bash
qrcheck run --ec Q --class 'boundary-*' --report-dir reports/boundary-q
qrcheck run --checks 'SY-*' --fail-fast --report-dir reports/symbol-only
```

Give each run its own `--report-dir`: a second run into the same root replaces the first run's three report files and its evidence tree, so each run should still get its own root.

`run` is quiet apart from one summary line on stdout: diagnostics go to stderr and every detail lands in the reports.

## What you can do with it

1. **Declare a target in configuration.** One `[targets.<name>]` table declares a generator as `library` (in-process import and call), `cli` (argv-only subprocess) or `http` (loopback request), with the mapping from framework fields to that target's inputs, outputs and metadata. No vendor endpoint, field name or response shape is hard-coded anywhere in the package.
2. **Verify target and toolchain before testing.** `qrcheck doctor` checks configuration validity, adapter resolution, one probe generate, both decoders, the SVG rasterizer and the platform wheels, and it separates a missing primary decoder — blocking — from a missing secondary decoder or rasterizer, which degrades visibly in the report instead.
3. **Generate a deterministic payload corpus.** `qrcheck corpus build --seed N` produces the 14 payload classes covering capacity boundaries, character classes, structured formats (URL, vCard, WiFi, MECARD, OTP) and pathological repeats, byte-identically from the seed.
4. **Run the suite at a chosen level and seed.** `qrcheck run --ec L|M|Q|H --seed N`, with `--checks`, `--class` and `--fail-fast` to narrow the work.
5. **Get per-check results with stable identity and evidence.** Every case yields a stable check ID and, when it fails, a derived failure ID; every failing case carries the image the target produced, the request that produced it and the decoded-or-error outcome.
6. **Consume three report formats.** JUnit XML for CI, `summary.json` for tooling, and one self-contained `index.html` for a human — it opens with no network access.
7. **Prove the suite is not vacuous.** `reference` must come back with zero failed, zero error and zero skipped; `defective` must fail exactly the nine checks its five injected defects target.
8. **Measure the framework against its own budget.** `qrcheck loc` counts tracked physical lines and gates on the ceiling, and every run records the same number as `loc_count`.

## Command surface

One global option precedes the subcommand: `qrcheck --config PATH <command>`, defaulting to `qrcheck.toml` in the current directory. Relative paths inside the file resolve against the file's own directory.

| Command | Options | What it does |
|---------|---------|--------------|
| `qrcheck run` | `--adapter`, `--ec L\|M\|Q\|H`, `--seed`, `--checks`, `--class`, `--fail-fast`, `--report-dir` | Runs the selected checks against one target and writes the three reports plus the evidence tree |
| `qrcheck doctor` | `--adapter`, `--json` | Verifies the configuration, the target, both decoders, the rasterizer and the platform before a run |
| `qrcheck list` | `--json` | Enumerates the registered checks with family, capabilities and optional dependencies |
| `qrcheck corpus build` | `--seed`, `--out`, `--force` | Regenerates the corpus from a seed and reports whether the file changed |
| `qrcheck loc` | `--limit`, `--path` | Prints per-file and total tracked physical line counts and gates on the limit |

`--checks` accepts comma-separated exact IDs (`RT-ROUNDTRIP`), ID globs (`SY-*`) and family selectors (`family:render`); a selection that matches nothing exits 2 and echoes the selection. `--class` is a repeatable glob over the corpus record's `class` field, so `--class 'boundary-*'` narrows a run to the boundary subset; a filter that leaves no record exits 2 naming the filter. `--adapter`, `--seed`, `--report-dir` and `--limit` default to the configured values. `--fail-fast` stops launching new checks after the first failed or errored one, lets the in-flight check finish, and still writes a schema-complete report — it shortens a run, never re-runs a case.

The two machine-readable variants exist for tooling: `list --json` lets a CI job compare the registry against the checks a run actually executed, and `doctor --json` turns a broken environment into a report a bug can quote. Every configuration key, its type, default, validation rule and exit-2 diagnostic lives in [`docs/configuration.md`](docs/configuration.md).

## Exit codes

| Code | Meaning | When it is produced |
|------|---------|---------------------|
| `0` | Run completed with `failed == 0` and `error == 0` | After the reports are written; skipped results are permitted and never count as a pass, because they are tallied separately |
| `1` | At least one check failed and none errored | After the reports are written |
| `2` | Configuration, selection, corpus-filter, corpus-file, an unusable pinned dependency — the primary decoder, the oracle or a core library, whether absent or installed but not importable — or a target-availability error | Before any check runs, so **no report is written**; the offending key, filter, file or dependency is named |
| `3` | A check errored internally, `report_dir` is not writable, or an unhandled framework exception occurred | After the reports are written — except for an unwritable `report_dir`: a run whose report cannot be written streams every result it holds to stdout as one JSON object per line, while a `report_dir` found unwritable before the run starts has no result to stream, so the diagnostic alone is written to stderr, naming the path and stating that nothing was streamed |

Precedence: exit 2 is decided before execution and so never competes with a result; during and after a run, 3 outranks 1, which outranks 0.

Per command: `doctor` uses 0 when every blocking prerequisite passes and 2 when one fails; `list` uses 0; `corpus build` uses 0, 2 for a bad argument and 3 for a write failure; `loc` uses 0 under the limit, 1 at or above it and 2 for a missing path.

## Configuration

[`qrcheck.toml`](qrcheck.toml) at the repository root is the committed default: `default_adapter = "reference"`, `report_dir = "reports"`, `seed = 20240101`, `corpus_path = "corpus/default.jsonl"`, `loc_limit = 60000`, the two built-in `[targets.*]` tables, and the `[render]`, `[checks.performance]` and `[checks.damage]` tables that own the thresholds summarised in the budgets table below. Validation is central, so an unknown key at any level, a `type` outside the five declared kinds, a `returns` or `image_encoding` value outside its enum, a syntactically invalid `callable`, a non-loopback `url` or a non-positive `timeout_s` each exits 2 naming the offending dotted key path. A capability shortfall is not a configuration error: a selected check whose requirements the target's declared capabilities cannot satisfy reports `skipped` with reason `unsupported_capability`, counted separately and never as a pass.

## Reports and evidence

- **`summary.json`** is the machine schema: a `run_id` that is a content hash over adapter, seed, EC level, selected check IDs and the corpus digest — never a UUID or a timestamp — the four counts `passed`, `failed`, `skipped` and `error` as separate integers so a skip can never be read as a pass, and one object per case with `check_id`, `case_id`, `status`, `reason`, `duration_ms`, `request`, `decoded`, `expected`, `evidence_dir` and `seed`, ordered by `(check_id, case_id)`.
- **`junit.xml`** is the framework's own writer in the `xunit2` shape: one `testcase` per case with `classname="qrcheck.<family>"` and `name="<CHECK-ID>[<case_id>]"`, a `skipped` element carrying the reason, a `failure` element carrying the decoded-versus-expected detail, and an `error` element carrying the traceback. No `hostname` and no `timestamp` is emitted, so the file is reproducible.
- **`index.html`** is one self-contained file: inline CSS, no script, no external stylesheet, font or image, every evidence image a `data:` URI, every target-supplied string escaped. It carries the counts and the run parameters in a header, a table of every result, one block per failing case with the image, the request, the outcome and the seed, and a separate block for skipped results with their machine-readable reasons. A banner states when the run degraded to one decoder or lost the rasterizer.

Evidence is written only for failing, errored and skipped-with-reason cases; a passing case records `evidence_dir: null`, so a full run does not produce tens of thousands of files. Every artefact lives under `<report_dir>/evidence/<check-id>/<failure-id>/`, where the failure ID is derived once from the check ID, the case ID and the reason and sanitized to `[A-Za-z0-9._-]`:

| Artefact | Written by | Content |
|----------|-----------|---------|
| `request.json` | every case with evidence | The request that produced the artefact, with the check ID, case ID, seed and resolved case parameters |
| `symbol.png` / `symbol.svg` | every image-producing check | The artefact the target returned, byte for byte |
| `decoded.json` | every case with evidence | Both decoder outcomes with their polarity, the agreement class, the EC-level pair, version, mask, matrix digest and the check's measurements |
| `damage-<kind>-<seed>.png` | `DM-RESILIENCE`, `DM-SEEDED` | One damaged image per variant, named with the derived seed that reproduces it |
| `symbol-repeat.png`, `logo.png` | `DT-DETERMINISTIC`, `RN-LOGO` | The second call's artefact, and the overlay bitmap that was composited |

## Determinism and reproduction

A run is fully determined by the adapter, the seed and the EC level: one derivation function turns the run seed into every corpus, occluder, noise and logo draw, iteration is sorted everywhere it reaches a report or an evidence listing, and no check's assertion reads a clock, the locale, the hostname or the filesystem's order. Durations are recorded but compared nowhere outside the two performance probes and the ceiling the end-to-end proof applies to the run as a whole.

`summary.json` therefore reproduces a run on its own — replay it with the recorded `--adapter`, `--ec` and `--seed`, and a single damaged image can be regenerated from the run seed, check ID, case ID and damage kind without re-running anything else. `DM-SEEDED` closes that loop from the other side: it re-applies the recorded damage to the symbol the case already holds, without calling the target again, and asserts the resulting image is byte-identical to the variant `DM-RESILIENCE` produced. To re-run one check rather than the suite:

```bash
qrcheck run --checks DT-DETERMINISTIC --seed 20240101
```

## Repository map

```text
qrcheck/          the package: cli, config, targets, decode, corpus, checks, registry, report, evidence
tests/            unit/ (contracts in isolation) and integration/ (real decoders, real transports, the e2e proof)
corpus/           default.jsonl — the committed 240-record corpus, 40 of them boundary payloads
examples/         cli_target.py and http_target_server.py — executable targets the adapter tests drive
docs/             six documents, indexed below
qrcheck.toml      the committed default configuration, including every threshold
pyproject.toml    build metadata, the console script, the seven runtime pins, pytest and coverage config
requirements.*    the two compile inputs and their two hash locks
.github/          workflows/ci.yml — the fast unit gate and the end-to-end proof
```

`reports/` is generated at run time and is not committed, so a report can never contain a stale duration.

## Documentation

| Document | Read it when you are |
|----------|----------------------|
| [`docs/architecture.md`](docs/architecture.md) | the implementer: the layers, the adapter port, the decode pipeline, the consensus rules, the oracle |
| [`docs/checks.md`](docs/checks.md) | reading a failure and asking what the check asserted and what its evidence proves |
| [`docs/configuration.md`](docs/configuration.md) | wiring a target: every key with its type, default, validation and exit-2 diagnostic |
| [`docs/targets.md`](docs/targets.md) | writing an adapter: the three configurable kinds, the placeholders, `returns`, `metadata_map`, capabilities |
| [`docs/performance.md`](docs/performance.md) | questioning a budget: the measurement environment, how each percentile is taken, what is gated |
| [`docs/dependencies.md`](docs/dependencies.md) | regenerating a lock or diagnosing a host where the tool will not start |

## Budgets and gates

| Budget | Declared once in | Read by |
|--------|------------------|---------|
| At most 60,000 tracked physical lines | `loc_limit` | `qrcheck loc` in CI, which takes the ceiling from `loc_limit` rather than restating it, and exits 0 only strictly under it |
| Encode p95 ≤ 250 ms | `[checks.performance] encode_p95_ms` | `PF-ENCODE` |
| Decode p95 ≤ 400 ms for a 500×500 PNG | `[checks.performance] decode_p95_ms` and `decode_image_px` | `PF-DECODE` |
| ≥ 120 symbols per minute | `[checks.performance] min_symbols_per_minute` | `PF-ENCODE` |
| Contrast ratio ≥ 7:1 | the per-target `contrast_floor` | `RN-CONTRAST`, and `IV-BAD-OPTION` when it builds a request below the floor |
| Logo overlay ≤ 15 % of symbol area | `[checks.damage] logo_area_fraction` | `RN-LOGO` |
| Occlusion ≤ the decoded EC capacity minus a safety margin | `[checks.damage] occlusion_margin` | `DM-RESILIENCE`, `DM-SEEDED` |
| Rotation ≤ 15°, perspective ≤ 10°, JPEG quality 40, downscale to 3 px per module | `[checks.damage]` | `DM-RESILIENCE`, `DM-SEEDED` |
| A full default run ≤ 10 minutes on 2 vCPU | derived from a run's own `started_at` and `finished_at`, with no configuration key | the end-to-end gate, which asserts the whole five-run proof against that ceiling and leaves the job timeout as the hard backstop |

Changing a threshold means changing the `qrcheck.toml` default and the criterion that reads it; no check, test or document carries a second copy of the number. A breached floor is a failed case rather than a warning, so it turns the affected check `failed`, makes the run exit 1 and fails the clean-run gate with it. [`ci.yml`](.github/workflows/ci.yml) runs the gates in two jobs: a fast unit job (syntax, `pytest -m "not e2e"` with the coverage floor, then the line-ceiling gate), and the end-to-end proof over both self-check targets and the boundary subset at L, Q and H, with the SVG rasterizer installed so `RN-SVG` is live rather than skipped.

## Scope boundaries

- **Camera or flatbed capture of printed symbols** — needs hardware and optical calibration; the framework tests generated bitmaps and vector output, not print-and-scan.
- **Micro QR and Structured Append** — generator support is inconsistent and reference-decoder metadata for them is unreliable, so Model 2 is the whole domain.
- **Stylised or artistic output** — custom module shapes and image-blended backgrounds are out; an explicit logo overlay at a declared EC level is the one compositing case that can be asserted.
- **A database, accounts or a hosted service** — the framework is a local CLI and its results are files.
- **PDF or EPS rendering** — PNG and SVG only; PDF would add a rasterization dependency without an extra assertion on symbol correctness.
- **A web UI** — the HTML report is static.
- **Retry, flaky quarantine or re-run machinery** — an intermittent result is a framework defect to fix, not to suppress.
- **Windows, and Python other than 3.11** — outside the platform contract; `doctor` reports an unsupported interpreter rather than pretending otherwise.
- **Load or soak testing beyond the throughput floor, and parallel or distributed execution** — the floor is the contract, and parallelism would break determinism and the 2-vCPU budget.
