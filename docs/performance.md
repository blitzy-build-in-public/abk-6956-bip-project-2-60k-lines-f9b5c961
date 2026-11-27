# qrcheck performance budgets

The four budgets `qrcheck` holds itself to — encode latency, decode latency, throughput and the wall-clock ceiling on a full proof — where each one is defined, how the two percentiles are measured, and what may never enter an assertion. Read this when you are challenging a number: why 250 ms, whether a sample count may rise, or why a run that looked slow on your machine still passed CI.

Every other subject has an owner elsewhere, and this document names it rather than repeating it: `checks.md` for what each check asserts and the per-check case matrix; `configuration.md` for the type, default, validation rule and exit-2 diagnostic of every configuration key; `dependencies.md` for the pins, the lock files and the platform resolution matrix; `architecture.md` for the pipeline, the layering and the no-retry rule; `targets.md` for declaring a target. No budget below is invented: each is the value the committed `qrcheck.toml` carries, and the key named beside it is where it is changed.

## 1. What the budgets are for, and what they are not

The budgets answer one question: does a full default suite finish inside a normal CI merge gate? A run at EC M over the 240-record corpus makes 1,287 encode invocations and 1,470 primary decodes (§5), so a pathological adapter, or a decode path that regressed by an order of magnitude, would turn a run that belongs in a merge gate into one nobody waits for. The floors bound the cost of one encode and one decode; the ceiling bounds the whole five-run proof.

They are not a benchmark of the target library. `PF-ENCODE` times the adapter call, whatever the adapter is — an in-process library, a subprocess or a loopback HTTP request — so its figure includes the target's own work. `PF-DECODE` never calls the target at all: it decodes an image the framework drew, so its figure is the decode pipeline's and a slow target cannot fail it. Neither probe says anything about the quality of a single symbol; symbol quality is what the other nineteen checks exist for.

The floors are the contract. Meeting them is the requirement, and optimising past them is out of scope: a change that makes the product faster without moving a declared budget changes nothing a gate measures, so it is not a reason to touch either probe.

## 2. The budget environment

A timing gate means nothing measured somewhere else, so the environment is named rather than assumed: **CPython 3.11** on Linux **x86_64** with **2 vCPU**, installed from the exact `requirements.lock` and `requirements-dev.lock` with `--require-hashes`. Those are the files CI installs, so the interpreter, the decoders, the oracle and the image libraries behind every figure are the pinned ones; `dependencies.md` owns the pins and the platform matrix.

The CI label is `ubuntu-latest`, which is 4 vCPU for a public repository and 2 vCPU for a private one, and measured wall-clock across the CPU models GitHub-hosted runners use varies by more than 2.5×. An absolute duration is therefore not portable, and the product does not treat one as if it were: every run records its own `started_at` and `finished_at` in `summary.json`, and the ceiling assertion reads those two fields, so a slow host produces a named duration to argue with instead of a job killed at a timeout.

The recorded span is the whole `run` invocation's work — the availability probe, the corpus read, the check walk and the line-count sweep that fills `loc_count` — not the sum of the case durations. That is deliberate: the ceiling is about how long a caller waits, not about how the case time was spent.

## 3. How the two percentiles are measured

Both probes live in `qrcheck/checks/performance.py`, and both have the same shape: a fixed number of untimed warm-up operations, then a fixed number of timed ones. The warm-up is what makes a reported percentile describe steady state instead of interpreter start-up. The counts are configuration, never literals — `warmup` (default `5`) precedes both probes, `encode_samples` (default `100`) is the encode window and `decode_samples` (default `200`) the decode window — and each probe reads a monotonic clock, `time.perf_counter`, so a difference can only produce a duration and never a date.

### 3.1 `PF-ENCODE` — encode latency and throughput

One payload, a declared ASCII constant short enough to sit inside the capacity of the smallest symbol at every error level, is encoded `warmup` times untimed and then `encode_samples` times timed, one clock reading per call. That is **105** target invocations (5 + 100) and **zero** decodes. From the samples the case computes the p95 (§3.3) and the throughput over the measured window — samples produced divided by the window's own elapsed seconds, times 60 — so the rate reported is the rate the samples were actually produced at: no warm-up inside it, no extrapolation from a mean.

Two floors are judged against that one measurement: encode p95 at or below `[checks.performance] encode_p95_ms` (default `250.0` ms), and throughput at or above `[checks.performance] min_symbols_per_minute` (default `120.0` symbols/minute). Every breached floor is reported, not only the first, so a host that is both slow and unproductive says so once, in one result, with each measured value beside its configured threshold. The check declares the `png` capability and the primary-decoder dependency, and it needs a symbol from the target even though it decodes nothing itself.

The invocation count is a promise rather than an observation: the module derives it from the configuration as `warmup + encode_samples` and publishes it as the check's expected adapter-invocation count, which `tests/unit/test_checks_negative.py` asserts with a counting stub. A retry of one slow sample would register as 106 and fail that test. There is no retry anywhere in the product (`architecture.md` §14), so a slow outlier becomes a reported p95 rather than a reason to measure again.

### 3.2 `PF-DECODE` — decode latency of one fixed probe image

`PF-DECODE` times `warmup` untimed decodes followed by `decode_samples` timed decodes of **one** PNG exactly `decode_image_px` × `decode_image_px` pixels (default `500` × `500`): **205** primary decodes and **zero** target encodes. The image is drawn inside the framework — the payload encoded by the oracle at the run's error level with `boost_error=False`, padded with the four-module quiet zone a decoder needs, then scaled by nearest neighbour so output pixel `i` takes source pixel `i * side // image_px` and module edges are preserved exactly. The framework owns that image, which is why the check declares no capability at all and stays runnable against a target that cannot encode, cannot render or cannot be reached.

Regenerating the image per sample would measure the wrong thing and waste the target's time. Decode dominates the cost of a case, so the probe holds the image fixed and varies only the number of decodes: a rising figure is then a statement about the decode path, not about the encoder that drew the probe. For the same reason the first decode happens before anything is timed and must return the payload the image was drawn from; a probe the pipeline cannot read fails the case instead of producing a percentile over failed decodes.

One floor is judged against that measurement: decode p95 at or below `[checks.performance] decode_p95_ms` (default `400.0` ms) for a PNG of `decode_image_px` (default `500`) pixels square. Both numbers travel in the result's `expected`, so a report states the probe size it measured as well as the floor it was judged against.

### 3.3 The percentile rule

Both p95 values are the **nearest-rank** percentile of the sample durations: the samples are sorted ascending and the reported value is the one at index `ceil(0.95 × n) − 1` — the 95th of the 100 sorted encode samples, the 190th of the 200 decode samples. The rule is stated because an undocumented percentile is not reproducible: nearest-rank, linear interpolation and an inverted CDF agree on no two data sets. No sample is discarded and no sample is re-taken, so two runs on the same host with the same lock measure the same statistic and a reader comparing two figures is comparing like with like.

### 3.4 Where the figures land in the report

Each probe is one result in `summary.json` `results[]`, identified by `check_id` (`PF-ENCODE`, `PF-DECODE`) and `case_id` (`synthetic:pf-encode`, `synthetic:pf-decode`). Its `expected` holds the thresholds and the measurement parameters the run applied, copied from the resolved configuration; its `decoded.metrics` holds what was measured.

| Result field | `PF-ENCODE` keys | `PF-DECODE` keys |
|--------------|------------------|------------------|
| `expected` | `encode_p95_ms`, `encode_samples`, `min_symbols_per_minute`, `warmup` | `decode_image_px`, `decode_p95_ms`, `decode_samples`, `warmup` |
| `decoded.metrics` | `encode_p95_ms`, `symbols_per_minute`, `samples`, `warmup` | `decode_p95_ms`, `image_px`, `samples`, `warmup` |

A passing probe writes no evidence, by the persistence policy of `qrcheck.report.evidence`. A failing one necessarily does, and its `decoded.json` carries the same `metrics` beside the request and the artefact: the measurement is the finding, and a failure whose evidence omitted the numbers would be unreadable. Those are the only places a latency figure is published. Per-case `duration_ms` exists in every row of `summary.json` and is one of the fields masked out of every determinism comparison (§7).

## 4. The four budgets, and the one owner of each

| Budget | Declared once in | Read by | Measured value lands in | Blocks |
|--------|------------------|---------|-------------------------|--------|
| Encode p95 ≤ **250 ms** | `[checks.performance] encode_p95_ms`, default `250.0` | `PF-ENCODE` | `results[].decoded.metrics.encode_p95_ms`, judged against `results[].expected.encode_p95_ms` | the `e2e` job, on every full run |
| Decode p95 ≤ **400 ms** for a **500 × 500** PNG | `[checks.performance] decode_p95_ms`, default `400.0`, with `decode_image_px`, default `500` | `PF-DECODE` | `results[].decoded.metrics.decode_p95_ms`, judged against `results[].expected.decode_p95_ms` | the `e2e` job, on every full run |
| **≥ 120 symbols/minute** | `[checks.performance] min_symbols_per_minute`, default `120.0` | `PF-ENCODE` | `results[].decoded.metrics.symbols_per_minute`, judged against `results[].expected.min_symbols_per_minute` | the `e2e` job, on every full run |
| A full five-run proof ≤ **600,000 ms** (ten minutes) | **derived — no configuration key** | `tests/integration/test_end_to_end.py` | `summary.json` `started_at` and `finished_at` of each of the five runs | the `e2e` job, with the job's `timeout-minutes: 30` as its hard backstop |

The run ceiling has no key by design. It is a property of the whole proof rather than of one check: `RunResult` carries the two timestamps and derives a duration from them, that derived value is deliberately absent from `summary.json`, and no check reads it — a configuration key would let one run be gated against a different total from the next while the acceptance criterion that matters still spoke of ten minutes. The assertion that exists sums the five runs' own recorded spans and requires the total to be at most 600,000 ms, naming the total and each run's contribution when it is not.

The provenance of both halves of that gate matters. The **measured** gate is the in-test assertion over the five runs; the `e2e` job's `timeout-minutes: 30` is a hard backstop that kills a hung job, and it is never what a passing or failing run is judged by — it reports no duration and names no cause. The fast layer has its own bound in the `unit` job's `timeout-minutes: 15`, which likewise only stops the job and asserts nothing.

A breached floor is a failed case, not a warning. `PF-ENCODE` or `PF-DECODE` reports `failed` with its measured value beside the configured one, the run's exit code becomes 1, and `tests/integration/test_reference_run.py` — the gate that a conforming target passes every check with zero failed, zero error and zero skipped — fails with it. An encode p95 of 300 ms therefore stops a merge at both points: the exit code and the clean-run gate, either of which is sufficient.

A probe cannot dodge its floor by skipping it either. `PF-ENCODE`'s `png` requirement turns a target that does not declare the capability into a `skipped` result with reason `unsupported_capability`, counted separately and never as a pass, and the clean-run gate forbids a skipped result outright on the reference run.

## 5. The measured workload

The counts make the budget knowable in advance, and they belong to the check definitions rather than to the report. At EC M over the default 240-record corpus, one full run is **1,283 cases, 1,287 encode invocations and 1,470 primary decodes** — 1,469 of those decodes against a target that rejects the over-capacity request, because that case then has no image to decode. The per-check rows that add up to those totals are the case matrix in `checks.md` §3.22 and are not repeated here. The two probes are a small part of the workload they sit in — 105 of the 1,287 encodes and 205 of the 1,470 decodes — so measuring does not materially lengthen the run being measured.

The end-to-end proof costs five runs — `reference` and `defective` at EC M, then the 40-record boundary subset at L, Q and H — about **3,700 encodes and 4,100 decodes** in total. The two full runs supply 2 × 1,287 encodes and at most 2 × 1,470 decodes; the three boundary runs supply the rest, carrying every corpus-driven check over 40 records instead of 240 while the fixed samples and the constructed cases keep their size. Budgeted at roughly **90 s for a full run and 220 s for the whole sequence**, that leaves a margin of about **2.7×** under the 600,000 ms ceiling (600 / 220 ≈ 2.7). Both figures are planning estimates rather than the gate: the three boundary runs are the cheaper ones, because every corpus-driven check walks 40 records instead of 240, and the assertion reads the measured total of the five runs and never either estimate.

Two consequences follow. A full run's cost is a bounded quantity rather than an open one: the fixed samples and constructed cases decide how many operations happen, and the floors bound how long one operation may take. And a narrowed run — `--checks`, `--class`, `--ec` — changes the totals, which is exactly why the ceiling is asserted on the fixed five-run sequence and never on an arbitrary invocation: the workload has to be the same one every time for a wall-clock assertion to mean anything, and a `summary.json` records what was actually run rather than what the matrix predicted.

## 6. Why macOS is recorded but not gated

The gates are defined on the Linux environment of §2, and CI executes them only there. A macOS run is fully supported: the same interpreter contract, the same hash-locked install, the same correctness gates, and the same two probes measuring and reporting `expected` and `decoded.metrics` in that run's `summary.json`. What does not carry over is the authority of a timing figure.

So a contributor on an Apple-silicon laptop whose run reports an encode p95 of 800 ms has a fact about their machine. The probe still asserts wherever the suite is run, so that figure appears as a failed `PF-ENCODE` case in their own report — but no CI job measures on macOS, so nothing turns it into a red build, and it is not evidence that the product missed a floor, because the floor was defined on a 2 vCPU Linux host. Before offering such a figure as the explanation for a floor, re-measure on the budget environment.

`doctor` inventories the interpreter as its `python` dependency — `missing` and blocking below 3.11, `degraded` for another implementation or a newer 3.x — which is what makes an unsupported interpreter visible before a run instead of as a puzzling measurement afterwards; the platform matrix itself is `dependencies.md`'s. The rule in one line: correctness gates apply on every supported platform, timing gates apply where they are defined.

## 7. What may never enter an assertion

Wall-clock time enters exactly two places in the product: the two probe checks and the 600,000 ms ceiling that `tests/integration/test_end_to_end.py` asserts. Everywhere else a duration is recorded and compared nowhere — `results[].duration_ms` exists so a reader can see where the time went, and no test, gate or determinism comparison reads it. The rest of this section protects that rule.

None of the following may enter an assertion or an artefact: wall-clock time outside `PF-ENCODE`, `PF-DECODE` and the ceiling; the locale, the encoding environment, `PYTHONHASHSEED` or the timezone; the hostname, the user name or the process ID; an absolute path; the order a library returns anything in; the filesystem's order; and image bytes produced in a different environment, because byte identity for a framework-drawn artefact is promised only inside the pinned lock — `dependencies.md` §11 owns where that promise stops.

The determinism rule that follows concerns the volatile fields the schema requires. Two `summary.json` files are compared only after masking `started_at`, `finished_at`, every `results[].duration_ms` and the `decoded.metrics` block of the two `PF-*` results; two `junit.xml` files only after masking the `time` attributes, because the writer deliberately emits no `hostname` and no `timestamp`, which leaves `time` as the only volatile content. The `metrics` block is on that list for the reason the opening of this section gives: what it holds — `encode_p95_ms`, `decode_p95_ms`, `symbols_per_minute` — is a measured latency or throughput, judged once against a configured floor and never against another run's measurement, so two reference runs differ there exactly as they differ in the durations, and a masking list that stopped at the durations would promise an equality it cannot deliver. `architecture.md` §11 states the general prohibition; this is the performance reader's half of it.

## 8. Budgets owned by other documents

| Budget | Declared once in | Read by |
|--------|------------------|---------|
| At most 60,000 tracked physical lines | top-level `loc_limit`, default `60000` | `qrcheck loc` in the CI `unit` job, which reads that key instead of passing `--limit`, so retuning the key moves the gate; the same number lands in `summary.json` `loc_count` |
| Contrast ratio at least 7 : 1 | per-target `contrast_floor`, default `7.0` | `RN-CONTRAST` asserts at or above it; `IV-BAD-OPTION` reads it to build a request below it |
| Logo overlay at most 15 % of symbol area | `[checks.damage] logo_area_fraction`, default `0.15` | `RN-LOGO` |
| Occlusion at most the decoded EC capacity minus a margin | `[checks.damage] occlusion_margin`, default `0.05` | `DM-RESILIENCE`, `DM-SEEDED` |
| Rotation at most 15°, perspective at most 10°, JPEG quality 40, downscale to 3 px per module | `[checks.damage] rotation_deg`, `perspective_deg`, `jpeg_quality`, `downscale_px_per_module` | `DM-RESILIENCE`, `DM-SEEDED` |
| 240 corpus records, 40 of them boundary records | `qrcheck/corpus/builder.py`; `corpus/default.jsonl` is the committed result | `RT-ROUNDTRIP`, and the L/Q/H boundary passes of `tests/integration/test_end_to_end.py` |

None of these is a latency figure, and none is measured the way §3 measures one: they are fractions, counts and geometry, decided arithmetically or read out of the symbol. They are listed only so a budget question ends in the right file. `configuration.md` documents each key in full — type, default, validation and the exit-2 diagnostic for a violation — and `qrcheck/loc.py` owns the counting rule behind the line budget, including what a physical line is and which directories the sweep prunes; this document restates neither.

One sentence on the line gate's boundary, because it is stricter than it looks: `qrcheck loc` exits 0 when the total is **under** `loc_limit` and 1 when it is **at or above** it, so passing `--limit 60000` means the tree is strictly under 60,000 tracked physical lines.

## 9. What is deliberately not automated

Two limits, stated as limits rather than claimed as coverage.

The *visual* quality of `index.html` is not machine-verified: layout, legibility, and whether a reader can find a row on a narrow viewport are human judgements. What is verified is the substance — the file is self-contained (no script, no external stylesheet, font or image, every evidence image a `data:` URI), target-supplied text is escaped, and every failing case carries a non-empty image.

The accuracy of the *documented* error-correction recovery capacities is a property of ISO/IEC 18004, not of this product: the figures — L 0.07, M 0.15, Q 0.25, H 0.30 of codewords — are approximate in the standard itself, and real tolerance can be lower, since a 1-L symbol corrects at most 2 of 26 codewords. That is why the damage budget carries an absolute safety margin — `[checks.damage] occlusion_margin`, subtracted from the documented capacity — instead of trusting the percentage, and why `DM-RESILIENCE` asserts survival against a budget below the documented capacity rather than deriving a precise tolerance and asserting equality with it.

## 10. Changing a budget

Four steps, in this order, all of them.

1. **Change the declared value.** A budget's authority is its key: the effective value in the committed `qrcheck.toml` — `[checks.performance]` for the three latency and throughput floors, `loc_limit` for the size ceiling, a target's `contrast_floor` for contrast — and, when the fallback that applies to a key the file omits must move with it, the matching `DEFAULT_*` constant in `qrcheck/config/models.py`, which the schema's table references by name. Nothing else in the product holds a number.
2. **Confirm nothing else must change.** The probe checks read the table at run time and copy each threshold into the result's `expected`, so no check module, test or report writer needs editing. If one did, the threshold would be duplicated, and this step is where the duplication is removed instead. The invocation counts follow the sample sizes on their own, because the expected encode and decode counts are derived as `warmup + encode_samples` and `warmup + decode_samples`.
3. **Re-measure on the budget environment.** Run the `e2e` job and read the observed value out of that run's own `summary.json` `results[]`. A figure from another machine is not a measurement of the floor.
4. **Only then adjust the criterion.** For a latency or throughput floor the criterion is the success criterion that reads it; for the size ceiling it is the `qrcheck loc` gate; for the run ceiling it is the 600,000 ms assertion in `tests/integration/test_end_to_end.py`. A criterion still naming the old number is a gate that no longer matches the product.

Raising a sample count is a budget change too, and it moves two things: the measured window, and the cost of the run, because `PF-ENCODE` makes `warmup + encode_samples` target calls and `PF-DECODE` makes `warmup + decode_samples` decodes. Check the workload of §5 and the ceiling before raising either. The sample floors — `encode_samples` and `decode_samples` at 10 or more — exist so a percentile is computed from enough observations to mean anything, and lowering a count to make a floor easier to meet trades a real assertion for a cheaper one.

No check, test or document may carry a numeric literal that a gate reads: the value lives in its key, so a relaxation is visible as a change to a declared default and to the criterion that reads it, which is what "every threshold is declared once" means in `architecture.md` §13. This document quotes each value so a reader can recognize it, and the quotation carries no authority — a threshold changed in `qrcheck.toml` without a matching change here is a documentation defect, and a threshold changed here without a configuration change alters nothing at all, because the gate reads the key.
