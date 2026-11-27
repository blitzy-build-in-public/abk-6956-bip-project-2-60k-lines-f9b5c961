# qrcheck checks

All twenty-one checks `qrcheck` registers: what each one asserts, what it needs from the target, how many cases it runs, what evidence a failure leaves behind, the reason code that failure carries, and the command that reproduces it. Read this when you are holding a failing or unexpected result and need to decide whether the target is wrong, the framework is wrong, or the run could not observe what the check wanted.

This document is the single owner of the per-check assertion table. `README.md` points here instead of restating it. Every other subject has an owner elsewhere, and this file names it rather than repeating it: `architecture.md` for the pipeline, the layering and the reason the two decoders are combined the way they are; `configuration.md` for the type, default and validation of every configuration key; `performance.md` for what the floors mean and how they are measured; `dependencies.md` for the pins and the lock files; `targets.md` for declaring a `library`, `cli` or `http` target.

The list of checks is the registry's, not this document's. `qrcheck/registry/check.py` holds the twenty-one identifiers, the nine families and the capability vocabulary, `qrcheck list` prints what the registry contains, and `qrcheck list --json` prints the same set machine-readably. Every check implements the one `CheckContract` type, is registered at import of its family module, and is executable from a run — a check that only a test could call does not exist, and `tests/integration/test_registry_reachability.py` is the gate that holds the printed set, the registered set and the executed set equal.

## 1. How to read a result

### 1.1 The four statuses

One status per case, and exactly one.

| Status | What it means | Counted in |
|--------|---------------|------------|
| `passed` | The check observed what it asserts, and the case carries no reason. | `counts.passed` |
| `failed` | The target's output contradicts the assertion. The result names the contradiction in its `reason` and the measurements behind it in its evidence. | `counts.failed` |
| `skipped` | The case could not be attempted at all: the target did not declare a capability the check needs, an optional runtime tool was absent, or the platform denies the observation. The `reason` says which. | `counts.skipped` |
| `error` | The framework or the check raised instead of reaching a verdict, or the target was lost for reasons the framework reports rather than judges. The traceback travels in the result. | `counts.error` |

**A `skipped` result is never a pass.** The four counts are four separate integers (`qrcheck/report/model.py`, `Counts`), a skipped result increments only `counts.skipped`, and the exit code is computed from `failed` and `error` alone — so a run whose checks could not be attempted is visibly different from a run whose checks succeeded. The conforming-target gate makes the same point from the other side: `qrcheck run --adapter reference` must produce zero `failed`, zero `error` **and** zero `skipped`, which is what stops a check from being made to pass by making it skip.

Precedence among the exit codes is fixed: during and after a run, an internal error outranks a failure, and a failure outranks a clean run. A configuration or selection fault is decided before any check runs and never writes a report.

### 1.2 The reason vocabulary

Every non-passing result carries exactly one reason; a passing case carries none. The vocabulary is closed at thirteen members of `Reason` in `qrcheck/report/model.py`, and nothing else may appear in a `summary.json`, a `junit.xml` message or an `index.html` row.

| Reason | Meaning | Accompanies |
|--------|---------|-------------|
| `unsupported_capability` | The target did not declare a capability the check requires, so no request was sent. Resolved before execution by `qrcheck/registry/resolve.py`. | `skipped` |
| `missing_dependency` | A declared optional runtime tool was not found for this run — the SVG rasterizer, for `RN-SVG`. | `skipped` |
| `unsupported_platform` | The observation itself cannot be made on this platform: `CL-ARTIFACTS` reports this for its handle half where neither `/proc/self/fd` nor `/dev/fd` exists, while its temporary-file half still asserts. | `skipped` |
| `decoder_disagreement` | Both reference decoders returned a payload and the two payloads differ. The framework records both and picks no winner. | `failed` |
| `format_info_disagreement` | The error-correction level read out of the symbol's format information contradicts the level the decoder reported. Both values are kept. | `failed` |
| `target_rejected` | The target refused a request the framework considers valid. On the `IV-*` probes a declared rejection is the expected outcome and the result is a pass instead; every other check fails with the target's text verbatim in the evidence. | `failed` |
| `target_unavailable` | The target could not be reached at all: the process could not be spawned, the connection was refused, the declared import path was missing. | `error` |
| `target_timeout` | The target did not answer within its configured timeout. | `error` |
| `contract_error` | A response arrived and was unusable: an empty body, undecodable base64, bytes no image library can open, or a field the configuration names and the response lacks. | `failed` |
| `value_mismatch` | The observation differs from what the check asserted — the general contradiction of a concrete expectation. | `failed` |
| `sampling_failed` | No module grid could be recovered from the symbol, so an assertion about modules, versions, masks or regions could not be made. Reported rather than converted into a verdict. | `failed` |
| `capacity_violation` | A symbol came back for a payload that cannot fit the pinned version and error level, which a conforming target must refuse. | `failed` |
| `internal_error` | A check or the framework raised. The traceback travels in the result and the run continues to the next check. | `error` |

`internal_error` is one member beyond the twelve the technical specification enumerates: the specification's "internal framework exception" state requires the result to name the fault, and borrowing a check's verdict for it would mislabel the cause. Each of the other twelve is spelled and used exactly as the specification writes it.

The five adapter errors a target may raise are a different vocabulary from these reasons, and `qrcheck/targets/base.py` owns them: `TargetUnavailable`, `TargetTimeout`, `TargetRejected`, `TargetContractError`, `TargetUnsupported`. No adapter lets a library exception escape unwrapped, and one function — `qrcheck/checks/support.py`'s translation — turns those five into the statuses and reasons above. Which reason an adapter error becomes is therefore a property of one place, not of eighteen checks.

### 1.3 Failure identifiers

A failing, errored or skipped case is filed under a failure ID, and that ID is the leaf of its evidence directory and the string a CI trend line keys on.

```
failure_id(check_id, case_id, reason) -> "<CHECK-ID>__<case_id>__<reason>"
```

Three rules, all fixed by `qrcheck/report/model.py`:

- **One definition.** `failure_id()` is the only function in the product that derives this identifier; the evidence store and `qrcheck/evidence/naming.py` read it, and no writer or check re-implements it.
- **One safe path segment.** Every character outside `[A-Za-z0-9._-]` becomes a single `_`, so a case identifier can never traverse out of the report directory. A `corpus:` prefix therefore renders as `corpus_`, and a damage case's `corpus:url-001@blur` renders as `corpus_url-001_blur`.
- **Bounded at 64 characters.** An identifier longer than that keeps its first 55 characters and gains an underscore plus the first eight hexadecimal characters of the `sha256` of the untruncated identifier, so the whole identifier — prefix and digest together — stays inside the bound that a filesystem path segment needs.

A passing case has no failure ID and no evidence directory: its `evidence_dir` is `null`. That is what keeps a full run of a few thousand cases from writing tens of thousands of files.

## 2. Case identity and ordering

### 2.1 Case identifiers

Every case has a stable identifier, and there are exactly three forms.

| Form | Used by | Example |
|------|---------|---------|
| `corpus:<record-id>` | A case driven by one record of the committed corpus: `RT-ROUNDTRIP`, `SY-VERSION-MIN`, `SY-EC-MATCH`, `SY-MASK`, `MD-MODE-OPTIMAL`, `MD-CHARSET`, `RN-QUIET-ZONE`, `RN-MODULE-SIZE`, `RN-SVG` | `corpus:wifi-004` |
| `corpus:<record-id>@<kind>` | A damage case: one corpus record under one damage kind (`DM-RESILIENCE`, `DM-SEEDED`) | `corpus:url-001@blur` |
| `synthetic:<name>` | A case the framework constructs rather than draws from the corpus | `synthetic:iv-overcapacity`, `synthetic:contrast-3-inverted`, `synthetic:logo-2-h`, `synthetic:matrix-v10-a`, `synthetic:determinism-1`, `synthetic:pf-encode`, `synthetic:cleanup` |

The corpus-driven forms are built by the shared support layer from the record the runner hands the check, so a record's `class` filter (`--class`) narrows them and nothing else. The `synthetic:` names are fixed literals in the owning check module, which is what lets a reader reproduce one case from this document alone.

Those nine checks, together with `DM-RESILIENCE` and `DM-SEEDED`, are the **corpus-driven** checks: they declare `corpus_driven=True`, which is what makes the corpus file a prerequisite of a run that selects one (AAP §0.3.4), and a selection of the remaining synthetic-only checks runs without a corpus file at all. A `--class` filter always reads the corpus, because its diagnostic is a statement about the records.

Two case identifiers are the runner's rather than a check's. `synthetic:target`: when a target is lost mid-run, the runner files exactly one error result per affected check under that identifier, with reason `target_unavailable` or `target_timeout`, rather than one error per corpus record — a dead target is one fact about the run, not hundreds; checks that need nothing from the target keep running and keep their own results. `synthetic:cases`: when a check's own case generator raises, so that none of its cases can run, the runner files that check's single `error` result under this identifier with the traceback in its evidence, and the walk continues with the next check — one broken generator never costs the report its other twenty checks (AAP §0.6.6 row 11).

### 2.2 Ordering

Order is part of the contract, because two runs' artefacts have to be comparable field by field.

- **Checks** are executed, listed and reported in ascending order of check identifier — the order of `CHECK_IDS` in `qrcheck/registry/check.py`, which is the ASCII order of the twenty-one strings — with one exception: `CL-ARTIFACTS` executes after every ordinary check, because its contract declares `observes_run_end` and its subject is the finished run (section 3.1), and its place in the artefacts is still its identifier's. Import order never reaches an artefact: `iter_checks()` is the only iteration a consumer may use and it sorts.
- **Cases within a check** are executed in ascending order of case identifier.
- **Results** in `summary.json`, `junit.xml` and `index.html` are ordered by `(check_id, case_id)`, so the three artefacts cannot disagree about the sequence.
- **Damage variants within a case** follow the declared kind order `occluders`, `blur`, `jpeg`, `noise`, `rotation`, `perspective`, `downscale` — a fixed tuple in `qrcheck/evidence/naming.py`, never a set's iteration order.
- **Every directory listing** a report or a test inspects is sorted before use, and every JSON artefact is written with sorted keys.

## 3. The twenty-one checks

One subsection per check, in check-ID order. Every subsection carries the same seven fields, and the vocabulary is the same in every one:

- **Requires** — the capability members of `Capability` (`qrcheck/targets/base.py`) the target must have declared for this check to be runnable. An unmet requirement is resolved before execution into `skipped` with reason `unsupported_capability`.
- **Optional runtime dependency** — the `DEPENDENCY_*` token the check declared, resolved once per run into `RunContext.availability`. A missing optional tool degrades to a skip; the primary decoder is never optional, because a run without it exits before any check.
- **Cases / encodes / decodes** — the row of the case matrix for one full run at error level M over the default 240-record corpus: the cases the check contributes, the `generate` invocations it makes and the decodes it performs.
- **Evidence** — names drawn only from the closed artefact set (`request.json`, `decoded.json`, `symbol.png`, `symbol.svg`, `symbol-repeat.png`, `damage-<kind>-<seed>.png`, `logo.png`), plus the measurements `decoded.json` carries for that check. Not every key of `decoded.json` is a measurement: a case whose target returned a symbol also records the target's own metadata claim under `declared_version`, `declared_ec`, `declared_mask` and `declared_mode` — evidence only, never asserted, `null` for a field the target did not state and absent when no symbol came back — and the same four keys appear in the decoded projection that case's own record carries in `summary.json`, its JUnit failure body and its block on the page.
- **Failure reasons** — the reasons this check's own assertion can produce. Every target-dependent check can *additionally* be skipped with `unsupported_capability`, fail with `target_rejected` when the target refuses a request the framework considers valid, fail with `contract_error` when the response is unusable, or receive one `synthetic:target` error with `target_unavailable` or `target_timeout` when the target is lost. Those five are properties of the shared translation in `qrcheck/checks/support.py` and of the runner, are stated once here rather than in eighteen rows, and a row names them again only where it changes the verdict.
- **Reproduce** — the command that re-runs this check alone, so the reader can confirm the finding without re-running the suite. Add `--report-dir <path>` when you do not want the re-run to overwrite the original report, and `--class <glob>` to narrow a corpus-driven check to the records you care about.

### 3.1 `CL-ARTIFACTS` — No temporary files or image handles left behind

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/cleanup.py` |
| Requires | none — this check is target-independent by contract |
| Optional runtime dependency | `primary-decoder` |
| Cases | one synthetic case, `synthetic:cleanup` |
| Cases / encodes / decodes | 1 / 0 / 0 |
| Evidence | `request.json` (identity only: the case observes the run, not the target) and `decoded.json` with `temp_prefix`, `temp_entries_before`, `temp_entries_after`, `leaked`, `fd_directory`, `handle_half`, `open_under_report_dir`, `open_image_count`, `open_images`, `violations` |
| Failure reasons | `value_mismatch`; `unsupported_platform` for the handle half alone |
| Reproduce | `qrcheck run --checks CL-ARTIFACTS --seed <seed>` |

**What it asserts.** That the run left nothing behind that a user could see. Two halves. The temporary-file half snapshots the entries of `tempfile.gettempdir()` whose name carries the framework's prefix `qrcheck-` before and after the run and requires the two snapshots to be equal — the runner takes the first one immediately before any check can run and executes this check after every other one, because the check declares `observes_run_end`, so the two snapshots bracket the run itself rather than this case alone; both are reported in `decoded.json` as `temp_entries_before` and `temp_entries_after`, and a `qrcheck-` entry any other case leaves behind fails it. That is also why it needs a temporary directory of the run's own (`TMPDIR`, `TMP` or `TEMP`): a sibling process leaving a `qrcheck-*` entry in a shared system temporary directory during the run is indistinguishable from this run's leak. The handle half enumerates `os.listdir('/proc/self/fd')` on Linux or `/dev/fd` on macOS — recorded in `decoded.json` as the token `proc-self-fd` or `dev-fd`, never as the directory's path — resolves every entry, and requires that no entry resolves to a path under the report directory; every image the framework opens goes through the `qrcheck/evidence/mux.py` context manager, so an unclosed handle is attributable to a case rather than merely counted. Where neither descriptor directory exists, the handle half reports `skipped` with reason `unsupported_platform` while the temporary-file half still asserts.

A failure here is a framework defect, not a target defect: no target can cause a leaked handle or a stray temporary file. If the finding is a leaked entry, the `leaked` list names it; if it is a handle, `open_images` names the case that opened it.

### 3.2 `DM-RESILIENCE` — Damage resilience: seven kinds of damage leave the payload readable

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/damage.py` |
| Requires | `png` |
| Optional runtime dependency | `primary-decoder` |
| Cases | eight sampled corpus records × seven damage kinds, `corpus:<record-id>@<kind>` |
| Cases / encodes / decodes | 56 / 8 / 64 |
| Evidence | `symbol.png` (the pristine artefact), one `damage-<kind>-<seed>.png` per variant, and `decoded.json` with `pristine` (`version`, `ec_level_decoder`, `ec_level_format`, `matrix_sha256`, `sampling`) and `damage[]` (`kind`, `seed`, `agreement`, `decoded_len`, `applied`, `per_decoder`) |
| Failure reasons | `value_mismatch`; `sampling_failed`; `contract_error` when the artefact the target returned is not an image any installed library can open, with its first 256 bytes in `decoded.json`; `target_rejected` and `unsupported_capability` through the shared adapter-error translation |
| Reproduce | `qrcheck run --checks DM-RESILIENCE --seed <seed> --ec <level>` |

**What it asserts.** That a symbol survives the damage its own error-correction level promises to survive. The sample is the first eight corpus records in `id` order whose payload is short enough to sit comfortably inside capacity, so the damage budget — not the payload size — is the only thing under test. Each sampled record is encoded once, and the seven variants are derived from that one symbol in the fixed kind order; the primary decoder must return the requested payload from every variant.

Those eight shared readings are why the row's decode count of 64 exceeds its 56 cases: each sampled record's pristine symbol is decoded once, memoised, and reused by all seven of its variants, because the occlusion budget is the recovery capacity of the error-correction level decoded from that symbol and of the module grid sampled from it — facts that exist only after a decode. The decode column of AAP §0.8.1's case matrix counts the 56 damaged reads alone.

Two rules shape the verdict. The occlusion budget is read from the **error-correction level decoded from the symbol**, never from the level that was requested, and the occluders are placed only in the data area — never over a finder, separator, timing or alignment pattern, a copy of the format information or the fixed dark module — because covering a function pattern makes a symbol undetectable however much capacity remains. And under damage only the primary decoder is authoritative: the secondary decoder's silence is recorded in `per_decoder` and never treated as a disagreement, which is what `architecture.md` describes as the consensus rules.

`sampling_failed` is reported when the pristine symbol's grid cannot be read at all, because then the budget the check would spend cannot be derived from the symbol. It is reported for one further reason, decided before any pixel is read: an artefact whose own header declares a canvas above the per-side ceiling this framework decodes is refused rather than allocated, and its declared dimensions are recorded in `decoded.json`.

### 3.3 `DM-SEEDED` — Damage reproducibility: the recorded seed regenerates the damaged artefact

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/damage.py` |
| Requires | `png` |
| Optional runtime dependency | `primary-decoder` |
| Cases | the same 56 `corpus:<record-id>@<kind>` identifiers as `DM-RESILIENCE` |
| Cases / encodes / decodes | 56 / 0 / 56 |
| Evidence | `damage-<kind>-<seed>.png` (the regenerated variant), and `decoded.json` with `damage_kind`, `damage_seed`, `applied`, `payload_sha256`, `payload_len`, `regenerated_sha256`, `recorded_sha256`, `regenerated_len`, `recorded_len`, `first_difference` |
| Failure reasons | `value_mismatch`; `sampling_failed` when the held artefact declares a canvas above the per-side ceiling this framework decodes; `missing_dependency` when the prerequisite is not available in this run; `contract_error` when a held artefact is not an image any installed library can open |
| Reproduce | `qrcheck run --checks DM-RESILIENCE,DM-SEEDED --seed <seed> --ec <level>` |

**What it asserts.** That the damage a run applied can be reproduced from what the report records, and nothing else. For each case it re-derives the seed through the product's single derivation point (`qrcheck/config/seeding.py`), re-applies the recorded damage to the symbol **it already holds**, and requires the result to be byte-identical to the variant `DM-RESILIENCE` produced in the same run. It never calls the adapter a second time — which is why its encode count is zero, and why a mask that changes between two calls cannot leak into this check.

The seed travels the same path as everything else that is reproducible: it is recorded in the result, in the damage filename and in `decoded.json`, so a single damaged image can be regenerated from `(run seed, check id, case id, damage kind)` without re-running anything else. `first_difference` reports the offset of the first byte that differs when the two do not match.

`missing_dependency` means the prerequisite is absent — `DM-RESILIENCE` was not part of this run, so no held symbol exists — not that a tool is missing. The memo those symbols live in belongs to one run: a second invocation in the same process starts with an empty memo and reports `missing_dependency` for all 56 cases whatever its target, seed and error level, so no run can compare its damage against another run's artefacts. Selecting both checks in one invocation is what makes `DM-SEEDED` a comparison at all, which is why the `Reproduce` command above names both.

### 3.4 `DT-DETERMINISTIC` — Identical requests produce identical symbols

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/determinism.py` |
| Requires | `png` |
| Optional runtime dependency | `primary-decoder` |
| Cases | six constructed cases, `synthetic:determinism-<n>` |
| Cases / encodes / decodes | 6 / 12 / 12 |
| Evidence | `symbol.png` (the first call), `symbol-repeat.png` (the second call), and `decoded.json` with the two per-call snapshots, `comparison` (`image_bytes`, `matrix_sha256`, `mask`), `divergent` and `unmeasured` |
| Failure reasons | `value_mismatch`; `sampling_failed` |
| Reproduce | `qrcheck run --checks DT-DETERMINISTIC --seed <seed>` |

**What it asserts.** That the same request produces the same symbol twice. Six payloads are each requested twice, and the two answers are compared component by component: the artefact bytes (`symbol_sha256`), the sampled module matrix (`matrix_sha256`) and the mask read from the format information. This is the only check that can see an encoder whose output varies between calls — each individual symbol is legal, decodable and in range, so no single-case assertion can catch it.

A component that was read and disagreed outranks one that could not be read, so an unsamplable second call can never hide a mask that changed between the two. Byte-identical results are promised within the pinned lock (`dependencies.md`), and never across environments.

### 3.5 `IV-BAD-OPTION` — An option outside the request contract is refused

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/invalid.py` |
| Requires | `png`, `ec_level`, `module_size`, `quiet_zone`, `color` |
| Optional runtime dependency | `primary-decoder` |
| Cases | four constructed probes: `synthetic:iv-bad-option-contrast`, `-ec`, `-module-size`, `-quiet-zone` |
| Cases / encodes / decodes | 4 / 4 / 0 |
| Evidence | `request.json` with `out_of_contract: true`, the case identity and the probe's request; `decoded.json` with the rejection text and `outcome`, `offending_field`, `value` (and, for the contrast probe, `contrast_ratio` and `contrast_floor`) |
| Failure reasons | `value_mismatch`; `contract_error` |
| Reproduce | `qrcheck run --checks IV-BAD-OPTION --seed <seed>` |

**What it asserts.** That the target refuses a request that breaks its own contract, and says which field broke it. Four probes, each breaking exactly one rule: an error-correction level outside `L`/`M`/`Q`/`H`, a module size of zero, a negative quiet zone, and a colour pair whose measured luminance ratio is ≈3.5 : 1 — below the floor the target itself declares, which is why the probe reads that target's `contrast_floor` rather than assuming a value.

A declared rejection is the expected outcome and the case **passes**, with the target's text kept verbatim. A symbol returned for a request that breaks the contract is `value_mismatch`. An untyped failure, a crash or a rejection the target cannot be recognised as having declared is `contract_error`.

These probes are deliberately illegal, so they cannot be built through the validating request builder: they come from `construct_out_of_contract()` in `qrcheck/checks/support.py`, and `request.json` marks them `out_of_contract: true` so no reader mistakes one for a request the framework considered valid.

### 3.6 `IV-EMPTY` — An empty payload is refused, or accepted as a documented success

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/invalid.py` |
| Requires | `png` |
| Optional runtime dependency | `primary-decoder` |
| Cases | one constructed case, `synthetic:iv-empty` |
| Cases / encodes / decodes | 1 / 1 / 0 |
| Evidence | `request.json` with `out_of_contract: true`; `decoded.json` with `outcome` (`declared_rejection`, `documented_success` or `incomplete_artefact`), `documented_success_accepted`, `payload_len` and, for an artefact, `output_format` and `symbol_len` |
| Failure reasons | `value_mismatch`; `contract_error` |
| Reproduce | `qrcheck run --checks IV-EMPTY --seed <seed>` |

**What it asserts.** That an empty payload produces one of exactly two acceptable outcomes: a declared rejection naming the empty payload, or a complete, documented artefact. What it must never produce is a crash, an unlabelled failure, or a partially written file — which is what `incomplete_artefact` names. Both acceptable outcomes pass, because whether an empty payload is an error is the target's contract, not the framework's opinion.

### 3.7 `IV-OVERCAPACITY` — A payload over the pinned version's capacity is refused

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/invalid.py` |
| Requires | `png`, `version_pin` |
| Optional runtime dependency | `primary-decoder` |
| Cases | one constructed case, `synthetic:iv-overcapacity` |
| Cases / encodes / decodes | 1 / 1 / 1 |
| Evidence | `request.json` with `out_of_contract: true`, `requested_len` and `capacity`; `symbol.png` when a symbol was returned; `decoded.json` with `requested_len`, `capacity`, `sizing_capacity`, `decoded_len`, `first_divergence_offset` and `per_decoder` |
| Failure reasons | `capacity_violation`; `contract_error`; `value_mismatch` |
| Reproduce | `qrcheck run --checks IV-OVERCAPACITY --seed <seed>` |

**What it asserts.** That a payload which cannot fit the pinned version and error level is refused rather than silently shortened. The probe is one byte above `max_payload_bytes(10, "M", "byte")` at a pinned Version 10, and it is deliberately built to exceed even the error level below the requested one, so a target that encodes at a lower level than asked cannot rescale the payload into an acceptable range. A declared rejection naming the capacity and the requested length passes; a symbol that comes back is `capacity_violation`, and the evidence shows `decoded_len` against `requested_len` with the offset of the first divergence.

This check builds its own payload outside the corpus, so no corpus-driven check observes the acceptance, which is why the defect it targets has no consequential failures.

### 3.8 `MD-CHARSET` — Charset preservation: every payload class round-trips byte for byte

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/mode.py` |
| Requires | `png` |
| Optional runtime dependency | `primary-decoder` |
| Cases | every record of the six charset coverages, `corpus:<record-id>` |
| Cases / encodes / decodes | 74 / 74 / 74 |
| Evidence | `symbol.png`; `decoded.json` with `payload_sha256`, `payload_len`, `expected_mode`, `first_divergence_offset` and the decoder projection |
| Failure reasons | `decoder_disagreement`; `value_mismatch` |
| Reproduce | `qrcheck run --checks MD-CHARSET --seed <seed> --ec <level> --class '<glob>'` |

**What it asserts.** That a payload survives the encode/decode round trip byte for byte whatever its character set, and that the class the corpus says it belongs to is the class that travelled. The six coverages map to the corpus classes as follows, and the mapping is asserted by `tests/unit/test_corpus_io.py` so no coverage can silently lose its records:

| Coverage | Corpus classes |
|----------|----------------|
| ASCII | `otp`, `boundary-alphanumeric` |
| UTF-8 multibyte | `utf8-multibyte` |
| Emoji | `emoji` |
| Right-to-left | `rtl` |
| Embedded NUL | `nul-byte` |
| Four-byte code points | `emoji`, plus the `utf8-multibyte` records that carry U+10000 and above |

The comparison is on bytes, never on rendered text, so a payload that is not valid UTF-8 is judged on what actually travelled. `first_divergence_offset` is what makes a truncation distinguishable from a substitution: the evidence shows the offset of the first byte at which the decoded payload departs from the requested one.

Two failing outcomes are decided in a fixed order, as the round-trip check decides them: two decoders whose payloads differ are a `decoder_disagreement` before they are a payload mismatch, because the framework does not pick a winner and a check that passed such a symbol while `RT-ROUNDTRIP` failed it would describe the same decode two ways; only a payload the authority really returned is then compared with the request, and a difference is a `value_mismatch` whose `first_divergence_offset` shows where it departed.

### 3.9 `MD-MODE-OPTIMAL` — Encoding mode is optimal: no symbol larger than the payload's own charset requires

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/mode.py` |
| Requires | `png` |
| Optional runtime dependency | `primary-decoder` |
| Cases | 24 discriminating boundary records plus a fixed 36-record byte-mode sample, `corpus:<record-id>` |
| Cases / encodes / decodes | 60 / 60 / 60 |
| Evidence | `symbol.png`; `decoded.json` with `expected_mode`, `max_version`, `payload_len_in_mode`, the decoded `version` and the decoder projection |
| Failure reasons | `sampling_failed`; `value_mismatch` |
| Reproduce | `qrcheck run --checks MD-MODE-OPTIMAL --seed <seed> --ec <level>` |

**What it asserts.** That the target chose a mode at least as compact as the payload's own character set admits. No decoder in the pinned pair reports a mode, so the assertion is made through the version: the version implied by the sampled module grid must be no larger than the smallest version the oracle needs when the payload is measured in its own charset. The discriminating half of the set is the whole of the two boundary classes — digit payloads that sit entirely inside the numeric charset and uppercase-and-symbol payloads that sit entirely inside the alphanumeric one — where the numeric or alphanumeric minimum is strictly below the byte-mode minimum; the byte-mode half is a fixed sample of records verified to lie outside both smaller charsets, where the same assertion proves the mirror image. This is the only way mode optimality is provable without mode metadata from either decoder (`architecture.md`).

### 3.10 `PF-DECODE` — Decode p95 latency of one fixed probe image against the configured floor

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/performance.py` |
| Requires | none — the check draws its own probe image and never asks the target for anything |
| Optional runtime dependency | `primary-decoder` |
| Cases | one constructed case, `synthetic:pf-decode` |
| Cases / encodes / decodes | 1 / 0 / 205 |
| Evidence | `decoded.json` with `metrics` (`decode_p95_ms`, `image_px`, `samples`, `warmup`) and the probe's own decode reading (`agreement`, `ec_level_decoder`, `ec_level_format`, `version`, `mask`, `matrix_sha256`, `payload_len`); `expected` carries the threshold, the sample count, the probe size and the warm-up that produced it |
| Failure reasons | `value_mismatch` |
| Reproduce | `qrcheck run --checks PF-DECODE --seed <seed>` |

**What it asserts.** That decoding one fixed probe image is fast enough. The probe image is drawn once by the framework — never by the target — and reused for every sample, so the reported percentile reflects steady-state decode rather than image generation; the untimed warm-up encodes and decodes precede the measured ones so interpreter start-up is not in the measurement. The threshold is `decode_p95_ms` in the `[checks.performance]` table of the configuration, the sample count is `decode_samples`, the warm-up is `warmup` and the probe side is `decode_image_px`; `performance.md` owns their meaning and the environment they are defined against. The probe must also decode, because a percentile over failed decodes would measure nothing.

The check declares no capability on purpose: it must stay runnable against a target that cannot encode, cannot render or cannot be reached, which is what makes it one of the two checks that still pass when a target is lost.

### 3.11 `PF-ENCODE` — Encode p95 latency and throughput against the configured floors

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/performance.py` |
| Requires | `png` |
| Optional runtime dependency | `primary-decoder` |
| Cases | one constructed case, `synthetic:pf-encode` |
| Cases / encodes / decodes | 1 / 105 / 0 |
| Evidence | `decoded.json` with `metrics` (`encode_p95_ms`, `symbols_per_minute`, `samples`, `warmup`); `expected` carries both thresholds, the sample count and the warm-up |
| Failure reasons | `value_mismatch` |
| Reproduce | `qrcheck run --checks PF-ENCODE --seed <seed>` |

**What it asserts.** That encoding is fast enough and productive enough: the p95 encode latency of one payload at or below `encode_p95_ms`, and the throughput at or above `min_symbols_per_minute`, over `warmup` untimed encodes followed by `encode_samples` timed ones. Both thresholds, the sample count and the warm-up live in the `[checks.performance]` table and are copied from the resolved configuration into the result's `expected`, so a run states the floor it measured against instead of a number the check module carries. `performance.md` owns what the floors mean, how p95 is computed and which environment they are defined for.

The measurement is one pass, never a retry: a slow outlier is reported as a p95, not smoothed away by re-running.

### 3.12 `RN-CONTRAST` — Contrast: the rendered palette matches the request and clears the declared floor

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/render.py` |
| Requires | `png`, `color` |
| Optional runtime dependency | `primary-decoder` |
| Cases | six palettes, each requested at both polarities: `synthetic:contrast-<n>-<polarity>` |
| Cases / encodes / decodes | 12 / 12 / 12 |
| Evidence | `symbol.png`; `decoded.json` with `contrast_ratio_measured`, `contrast_floor`, `palette_measured`, `palette_requested`, `polarity_requested`, `inverted_request` and the decoder projection including `polarity_applied` |
| Failure reasons | `value_mismatch`; `sampling_failed` |
| Reproduce | `qrcheck run --checks RN-CONTRAST --seed <seed>` |

**What it asserts.** That the target rendered the colours it was asked for, that the rendered pair clears the luminance ratio the target itself declares in `contrast_floor`, and that the symbol still decodes — at both polarities. The inverted half is why the check exists: a light-on-dark request must decode through the framework's polarity-normalised second pass rather than depending on a decoder feature that the pinned decoders do not both have, and `polarity_applied` records which pass produced the payload (`architecture.md`).

A failure with `value_mismatch` and a low `contrast_ratio_measured` means the target rendered a palette it cannot be read from; a failure with matching colours means the ratio was below the floor the target declared.

### 3.13 `RN-LOGO` — Logo overlay: the composite is applied and survives its error-correction budget

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/render.py` |
| Requires | `png`, `logo_overlay`, `ec_level`, `module_size` |
| Optional runtime dependency | `primary-decoder` |
| Cases | three overlay sizes, each at error level H and error level L: `synthetic:logo-<step>-<level>` |
| Cases / encodes / decodes | 6 / 6 / 6 |
| Evidence | `symbol.png`, `logo.png` (the seeded overlay bitmap that was composited, so the covered area can be re-measured), and `decoded.json` with `decoded_ec_level`, `logo_area_fraction_requested`, `logo_area_fraction_measured`, `logo_area_fraction_measured_image_area`, `logo_area_fraction_basis`, `logo_side_px`, `logo_seed`, `module_px_measured`, `occlusion_margin`, `recovery_fraction`, `recovery_budget`, `assertion_applied`, `rendered_colour_count`, `symbol_module_count` and `symbol_px` |
| Failure reasons | `value_mismatch`; `sampling_failed` |
| Reproduce | `qrcheck run --checks RN-LOGO --seed <seed>` |

**What it asserts.** That the overlay was actually drawn, at the size that was asked for, and that the symbol survives it wherever the symbol's own error-correction level says it should. The overlay bitmap is derived from the run seed and recorded as `logo.png`, so the covered area is measurable from the evidence rather than taken on trust; the three sizes are fractions of the configured `logo_area_fraction` ceiling.

The decode assertion is conditional by design: it is made only when the covered fraction is at or below the budget derived from the **error-correction level decoded from the symbol** minus the configured `occlusion_margin`. At error level H, the largest configured overlay sits inside that budget and the decode is asserted; at error level L it sits outside it, so the check records what happened and does not assert it — a logo that a low-redundancy symbol cannot survive is not a target defect. `assertion_applied` states which of the two readings the verdict used, and `recovery_budget` states the budget it was compared against.

### 3.14 `RN-MODULE-SIZE` — Module size: the rendered module edge matches the requested pixel size

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/render.py` |
| Requires | `png`, `module_size`, `quiet_zone` |
| Optional runtime dependency | `primary-decoder` |
| Cases | the first twelve corpus records in `id` order, `corpus:<record-id>` |
| Cases / encodes / decodes | 12 / 12 / 0 |
| Evidence | `symbol.png`; `decoded.json` with `module_px_measured`, `module_size_px_requested`, `module_px_deviation`, `module_px_tolerance`, `modules_across_canvas`, `quiet_zone_modules_requested`, `implied_module_count`, `implied_version`, `image_px` |
| Failure reasons | `sampling_failed`; `value_mismatch` |
| Reproduce | `qrcheck run --checks RN-MODULE-SIZE --seed <seed> --class '<glob>'` |

**What it asserts.** That a module is the number of pixels the request asked for, and that the image dimension obeys the dimension contract `(modules + 2 × quiet_zone) × module_size` for the quiet zone the check pins. Both are measured arithmetically from the image — the module pitch is taken from the top-left finder pattern's dark run — so no decode is needed and a symbol that no decoder can read still yields a verdict on its geometry; that is why the decode count is zero. The tolerance is the single named constant `MODULE_SIZE_TOLERANCE_PX` in the owning module, not a configuration key.

`sampling_failed` means no finder run could be measured in the artefact at all, so there is no pitch to compare.

### 3.15 `RN-QUIET-ZONE` — Quiet zone: every rendered margin is at least the requested modules

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/render.py` |
| Requires | `png`, `quiet_zone` |
| Optional runtime dependency | `primary-decoder` |
| Cases | the first twelve corpus records in `id` order, `corpus:<record-id>` |
| Cases / encodes / decodes | 12 / 12 / 0 |
| Evidence | `symbol.png`; `decoded.json` with `quiet_zone_measured` (`top`, `right`, `bottom`, `left`), `quiet_zone_requested`, `quiet_zone_tolerance_modules`, `module_px_measured`, `image_px` |
| Failure reasons | `sampling_failed`; `value_mismatch` |
| Reproduce | `qrcheck run --checks RN-QUIET-ZONE --seed <seed> --class '<glob>'` |

**What it asserts.** That the light margin on all four sides of the symbol is at least as wide as the quiet zone the request asked for. The margin is measured arithmetically from the image — the all-light module rows and columns outside the dark module bounding box, converted to modules through the measured pitch — so a quiet zone that is too short is reported even on a symbol no decoder can read; the verdict never depends on a decoder, which is why the decode count is zero. The requested zone cycles over a declared list of values so a target that ignores the key is caught, and the tolerance is the single named constant `QUIET_ZONE_TOLERANCE_MODULES` in the owning module.

`quiet_zone_measured` is the field to open first: it gives the four margins in modules beside `quiet_zone_requested`, so a short zone is visible as arithmetic rather than inferred from a message.

### 3.16 `RN-SVG` — SVG: the vector canvas carries the module count and rasterizes to a decodable symbol

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/render.py` |
| Requires | `svg`, `quiet_zone` |
| Optional runtime dependency | `svg-rasterizer` |
| Cases | the first twelve corpus records in `id` order, `corpus:<record-id>` |
| Cases / encodes / decodes | 12 / 12 / 12 |
| Evidence | `symbol.svg`; `decoded.json` with `rasterizer` (`configured`, `name`, `available`, `attempted`), `svg_bytes`, `svg_canvas_side`, `svg_canvas_unit`, `svg_module_count`, `svg_module_pitch`, `payload_matches_request`, `rasterized_bytes`, `problems`, and the raster's own decode projection |
| Failure reasons | `missing_dependency` (skip); `sampling_failed`; `value_mismatch` |
| Reproduce | `qrcheck run --checks RN-SVG --seed <seed> --class '<glob>'` |

**What it asserts.** Three things about vector output, because an SVG's byte layout is the encoder's business and cannot be asserted: that the document parses; that its canvas implies the module count the decoded version implies, with a module pitch that is positive and — for a plain user-unit canvas — a whole number of units; and that rasterizing the document yields an image the primary decoder reads — a raster the primary decoder cannot read is a failing case, `value_mismatch` with `problems: ["raster_does_not_decode"]`, while the decoded payload's agreement with the request is recorded in `decoded.json` as `payload_matches_request` rather than asserted. A canvas carrying a physical unit is held only to a positive pitch, because one module may lawfully be a fraction of a millimetre. The raster is composited onto white before decoding, since a vector canvas usually has no background rectangle.

The artefact is bounded before any of that is spent on it, because it is the target's bytes: a document longer than the module's declared byte ceiling, or one carrying a `<!DOCTYPE` declaration, an `<!ENTITY` declaration or an `<?xml-stylesheet` instruction, is refused before it is parsed — the parser expands an internal entity as it reads, so a few hundred bytes can become megabytes of text ahead of any geometry guard, and it drops a processing instruction rather than resolving it, so the stylesheet such an instruction names is refused from the raw bytes instead. Such a refusal is a `value_mismatch` recording `problems: ["svg_canvas_unreadable"]` and the refused declaration in `svg_error`. The document's references are restricted to same-document `#fragment`s: an `href` or `xlink:href`, and a CSS `url(...)` in any attribute value, must name a fragment of the document that carries it; an `@import` has no same-document form at all and is refused wherever it appears; and a reference to anything else is refused as `value_mismatch` with `problems: ["svg_external_reference"]`, quoting the value it refused and never reaching the tool, because the rasterizer would otherwise fetch a host this framework never chose or read a local file. A reference spelled as a CSS escape is refused the same way — `u\72 l(...)` and `\75 rl(...)` are one `url(...)` to a renderer, and a presentation attribute such as `fill` carries the same value a `style` attribute would — since that spelling names a resource the value scan cannot read and a conforming encoder emits no backslash anywhere. A declared canvas above the framework's per-side raster ceiling is then reported as `value_mismatch` with `problems: ["canvas_above_raster_ceiling"]` and the priced side in `svg_canvas_side_px`, never raised, because what a target's artefact says about itself is a verdict about that target and leaves the run's exit code at 1. Only a rasterizer or host fault — a binary that exits non-zero, times out or writes nothing — is an `error` result. A rasterizer that floods one of its output streams is a fault of that same kind: each stream is captured under a declared byte bound, so a stream that passes it is an `error` result naming the stream and the ceiling rather than growth in the runner's memory from a document that made the tool write more than it declared.

When no rasterizer can be found, `qrcheck/registry/resolve.py` resolves `RN-SVG` to one whole-check `skipped` result with reason `missing_dependency` before any case runs; that result records the discovery order it attempted in `decoded.attempted` and persists its evidence like any other skip, so the report names what to install rather than merely reporting an absence. Should the check meet an absent rasterizer in its own case enumeration — the per-case half of the same gate — each case is `skipped` the same way with the order in `rasterizer.attempted` and the status in `rasterizer`. `dependencies.md` owns that discovery order and the binary that satisfies it. Discovery runs once per run, in the check's case enumeration, so all twelve cases report the same rasterizer.

### 3.17 `RT-ROUNDTRIP` — Payload round-trip: the decoded bytes equal the requested bytes

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/roundtrip.py` |
| Requires | `png` |
| Optional runtime dependency | `primary-decoder` |
| Cases | every corpus record, `corpus:<record-id>` |
| Cases / encodes / decodes | 240 / 240 / 240 |
| Evidence | `symbol.png`; `decoded.json` with `payload_sha256`, `payload_len`, `decoded_len`, `first_divergence_offset` and the full decoder projection (`per_decoder`, `polarity_applied`, `agreement`, both EC readings, `version`, `mask`, `matrix_sha256`) |
| Failure reasons | `decoder_disagreement`; `value_mismatch` |
| Reproduce | `qrcheck run --checks RT-ROUNDTRIP --seed <seed> --ec <level> --class '<glob>'` |

**What it asserts.** That the payload the decoder recovers is the payload that was asked for, byte for byte, for every record of the corpus at the run's error level. The comparison is on bytes, never on text. Three failing outcomes are decided in a fixed order: two decoders whose payloads differ are a `decoder_disagreement` before they are a payload mismatch, because the framework does not pick a winner; no payload at all from the authoritative decoder is a `value_mismatch`, because there was nothing to compare; and a payload that differs from the request is a `value_mismatch` whose `first_divergence_offset` shows where it departed.

The primary decoder is the sole authority for this and every other check. A case the secondary decoder alone could read is still a failure, and the secondary payload is recorded in `decoded.json` as corroboration or contradiction rather than as a rescue — the consensus rules behind that are in `architecture.md`.

### 3.18 `SY-EC-MATCH` — Error-correction level from the decoder and from the format information matches the request

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/symbol.py` |
| Requires | `png`, `ec_level` |
| Optional runtime dependency | `primary-decoder` |
| Cases | every corpus record, `corpus:<record-id>` |
| Cases / encodes / decodes | 240 / 240 / 240 |
| Evidence | `symbol.png`; `decoded.json` with `ec_level_decoder`, `ec_level_format`, `request_ec_level`, `sampled_version` and `per_decoder` |
| Failure reasons | `decoder_disagreement`; `format_info_disagreement`; `sampling_failed`; `value_mismatch` |
| Reproduce | `qrcheck run --checks SY-EC-MATCH --seed <seed> --ec <level> --class '<glob>'` |

**What it asserts.** That the symbol carries the error-correction level that was requested, established from two independent sources: the level the primary decoder reports, and the level decoded out of the 15-bit format information inside the symbol. A symbol the consensus layer classifies `disagree` fails first, with `decoder_disagreement`, before either reading is compared: the classification describes what the decoders did, which is settled before any field of the symbol is judged. The three field-level outcomes are distinct on purpose. The two sources disagreeing with each other is `format_info_disagreement`, and both values are kept verbatim; they agreeing on a level that is not the requested one is `value_mismatch`; either reading being absent is `sampling_failed`, because nothing can be concluded.

This is the check that catches a generator which quietly encodes at a level other than the one it was asked for, and the evidence is what makes the finding diagnosable: `ec_level_decoder` and `ec_level_format` sit beside `request_ec_level` in one object, so the reader sees all three values without re-running anything.

### 3.19 `SY-MASK` — Mask pattern from the format information is legal and honours a pin

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/symbol.py` |
| Requires | `png` |
| Optional runtime dependency | `primary-decoder` |
| Cases | every corpus record, `corpus:<record-id>` |
| Cases / encodes / decodes | 240 / 240 / 240 |
| Evidence | `symbol.png`; `decoded.json` with `mask` (from the format information), `mask_pin`, `sampled_version` and `per_decoder` |
| Failure reasons | `decoder_disagreement`; `sampling_failed`; `value_mismatch` |
| Reproduce | `qrcheck run --checks SY-MASK --seed <seed> --ec <level> --class '<glob>'` |

**What it asserts.** That the symbol's mask is a legal mask, and that a mask that was pinned was honoured. No decoder in the pinned pair reports a mask, so it is read from the format information alone. Mask *selection* is encoder-defined (`architecture.md`), so an unpinned request is held only to the legal range `0..7` — asserting one value would fail a conforming target — and equality is asserted only when the request carried a pin. A symbol the consensus layer classifies `disagree` is a `decoder_disagreement` ahead of the mask reading, so a check here cannot pass a symbol the round-trip check fails.

### 3.20 `SY-MATRIX-EQUIV` — Module matrix equals the oracle at a pinned version, error level and mask

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/symbol.py` |
| Requires | `png`, `ec_level`, `version_pin`, `mask_pin` |
| Optional runtime dependency | `primary-decoder` |
| Cases | eight constructed cases at Versions 1, 2, 7 and 10, two payloads each of at most 64 bytes: `synthetic:matrix-v<version>-<label>` |
| Cases / encodes / decodes | 8 / 8 / 8 |
| Evidence | `symbol.png`; `request.json` carrying `expected_module_count` and `oracle_matrix_sha256` beside the request; `decoded.json` with `expected_module_count`, `expected_version`, `oracle_matrix_sha256`, `request_ec_level`, `sampled_module_count`, `sampled_matrix_sha256`, `sampled_mask`, `sampled_version`, `sampled_version_from_dimension`, `differing_modules` and `first_difference` |
| Failure reasons | `decoder_disagreement`; `sampling_failed`; `value_mismatch` |
| Reproduce | `qrcheck run --checks SY-MATRIX-EQUIV --seed <seed> --ec <level>` |

**What it asserts.** That the symbol's module matrix is the oracle's module matrix, module for module, at a pinned version, error level and mask. The pinning is the whole assertion: mask selection is a property of the encoder, so two encoders are only required to agree when every parameter that decides the matrix has been fixed, and this check therefore uses eight small payloads at four versions instead of the corpus. The failure paths are checked in order — a symbol the consensus layer classifies `disagree` is `decoder_disagreement`, ahead of every comparison below; no sampled grid is `sampling_failed`; a module count that is not the pinned version's is `value_mismatch`; a grid whose dimension implies a version other than the pinned one is `value_mismatch`; and any differing module is `value_mismatch` with `differing_modules` counting them and `first_difference` giving the row and column of the first.

To read a failure here, open `request.json` first: it holds the oracle digest and the expected module count, so the expectation can be regenerated from what the evidence records without reading the check's source. Then open `decoded.json` and compare `sampled_matrix_sha256` and `differing_modules` with it.

On the `defective` target this check fails too, and section 4 says why that is a declared consequence rather than a second defect: that target encodes at one level below the one it declares, so a matrix at a pinned error level necessarily differs from the oracle's. The broader limitation is stated plainly in `architecture.md`: the `reference` target is built on the oracle, so against it this check proves the render → sample → compare path rather than the independence of two encoders, and the `defective` target is what breaks the tautology.

### 3.21 `SY-VERSION-MIN` — Version is the smallest symbol that fits the payload, read from the module grid

| Field | Value |
|-------|-------|
| Family module | `qrcheck/checks/symbol.py` |
| Requires | `png`, `ec_level` |
| Optional runtime dependency | `primary-decoder` |
| Cases | every corpus record, `corpus:<record-id>` |
| Cases / encodes / decodes | 240 / 240 / 240 |
| Evidence | `symbol.png`; `decoded.json` with `expected_version`, `oracle_version`, `module_count_for_expected_version`, `request_ec_level`, `sampled_version` and `per_decoder` |
| Failure reasons | `decoder_disagreement`; `sampling_failed`; `value_mismatch` |
| Reproduce | `qrcheck run --checks SY-VERSION-MIN --seed <seed> --ec <level> --class '<glob>'` |

**What it asserts.** That the symbol is no larger than it needs to be. Neither decoder in the pinned pair reports a version, so the version is recovered from the symbol itself: the sampled grid dimension `N` implies the version through `N = 4V + 17`, and that version must equal the smallest version the oracle needs for the payload at the requested error level — or the version that was pinned, when the request carried a pin, because honouring a pin is the target's contract and not a size defect. `oracle_version` and `sampled_version` sit together in the evidence, so a symbol that is one version too large is visible as a number rather than as a message. A symbol the consensus layer classifies `disagree` is a `decoder_disagreement` before the sampled version is compared at all.

`ec_level` is a required capability for a reason: the smallest fitting version depends on the level, so a target that cannot be asked for a level cannot be held to a version.

### 3.22 The full-run workload

The counts above are the rows of the case matrix, and they add up to a fixed workload — which is what makes the run-duration budget and the expected totals of a conforming run knowable in advance.

| Check | Cases | Encodes | Decodes |
|-------|------:|--------:|--------:|
| `CL-ARTIFACTS` | 1 | 0 | 0 |
| `DM-RESILIENCE` | 56 | 8 | 64 |
| `DM-SEEDED` | 56 | 0 | 56 |
| `DT-DETERMINISTIC` | 6 | 12 | 12 |
| `IV-BAD-OPTION` | 4 | 4 | 0 |
| `IV-EMPTY` | 1 | 1 | 0 |
| `IV-OVERCAPACITY` | 1 | 1 | 1 |
| `MD-CHARSET` | 74 | 74 | 74 |
| `MD-MODE-OPTIMAL` | 60 | 60 | 60 |
| `PF-DECODE` | 1 | 0 | 205 |
| `PF-ENCODE` | 1 | 105 | 0 |
| `RN-CONTRAST` | 12 | 12 | 12 |
| `RN-LOGO` | 6 | 6 | 6 |
| `RN-MODULE-SIZE` | 12 | 12 | 0 |
| `RN-QUIET-ZONE` | 12 | 12 | 0 |
| `RN-SVG` | 12 | 12 | 12 |
| `RT-ROUNDTRIP` | 240 | 240 | 240 |
| `SY-EC-MATCH` | 240 | 240 | 240 |
| `SY-MASK` | 240 | 240 | 240 |
| `SY-MATRIX-EQUIV` | 8 | 8 | 8 |
| `SY-VERSION-MIN` | 240 | 240 | 240 |
| **Total per full run at EC M** | **1,283** | **1,287** | **1,470** |

One row carries a decode count larger than its case count: `DM-RESILIENCE`'s 64 is its 56 damaged reads plus one memoised pristine decode for each of the eight sampled records, because that pristine reading is what yields the error-correction level the occlusion budget is derived from and it is shared by all seven of the record's variants. The decode column of AAP §0.8.1's case matrix counts only the 56 damaged reads, which is where its 1,462 total comes from; the total above is that 1,462 plus the eight. One row's decode is conditional rather than fixed: `IV-OVERCAPACITY` has an image to decode only when the target accepts the over-capacity request, so a conforming `reference` run performs 1,469 decodes while a target that accepts it, such as `defective`, performs the full 1,470.

The samples are deterministic subsets rather than arbitrary lists. `RN-QUIET-ZONE`, `RN-MODULE-SIZE` and `RN-SVG` take the first twelve records in `id` order. `DM-RESILIENCE` and `DM-SEEDED` take the first eight records in `id` order whose payload length is between 8 and 64 bytes, which is short enough that no payload-truncation defect can reach them and comfortably inside capacity, so the damage budget is the only thing under test. `MD-MODE-OPTIMAL`'s twenty-four discriminating cases are the two boundary classes in full and its thirty-six byte-mode cases follow the same first-in-order rule over records verified to lie outside both smaller charsets.

A narrowed run (`--checks`, `--class`, `--ec`) changes these totals, which is exactly why the counts belong to the check definitions and not to the report: a `summary.json` records what was actually run, and the matrix above is what a full run at EC M produces.

## 4. The `defective` target: the nine checks that must fail

A suite that cannot fail proves nothing. `qrcheck` therefore ships two built-in self-check targets. `reference` is a conforming segno-backed generator and is the clean-run gate. `defective` is the same generator with five injections switched on, and it is the proof that the checks detect real defects rather than describing themselves. Both are ordinary adapters as far as the framework is concerned: no check knows either exists, and `architecture.md` describes the port they sit behind.

### 4.1 The five injections

The table is `DEFECTS` in `qrcheck/targets/defective.py`, and each injection is the minimum change its own check detects. Where a consequence cannot be avoided it is declared in the same entry rather than argued away or hidden by weakening a check.

| Injection | What the code does | Caught by | Declared consequential failures | Why the consequence is unavoidable |
|-----------|--------------------|-----------|--------------------------------|------------------------------------|
| `wrong_ec_level` | Encodes at one error-correction level below the requested one (`L`→`L`, `M`→`L`, `Q`→`M`, `H`→`Q`) while declaring the requested level in the returned metadata. `L` maps to itself because no level lies below it. | `SY-EC-MATCH` | `SY-VERSION-MIN`, `SY-MATRIX-EQUIV` | A lower level fits the same payload in a smaller version at a capacity boundary, so the minimal-version assertion must fail too; and the module matrix at a pinned error level necessarily differs from the oracle's. |
| `truncated_payload` | Encodes only the first 64 bytes of a payload longer than 64 bytes when the full length selects the same version at the level actually encoded, leaving intact any payload whose truncation would cross a version boundary. | `RT-ROUNDTRIP` | `MD-CHARSET` | With the window fixed this way the truncated payload is shorter but still occupies the same version, so `SY-VERSION-MIN` keeps the error-level shift as its single cause; `MD-CHARSET` fails because its multibyte, binary and alphanumeric payloads are longer than the window. |
| `missing_quiet_zone` | Renders a quiet zone of one module when four or more were requested — never zero — and sizes the image from the zone it actually drew. | `RN-QUIET-ZONE` | `RN-MODULE-SIZE` | The dimension contract is `(modules + 2 × quiet_zone) × module_size`, so a short zone necessarily breaks it. Stopping at one module rather than zero keeps the primary decoder able to read the symbol, and both assertions are measured arithmetically from the image, so neither needs a decode. |
| `non_deterministic_mask` | Cycles the mask per call (`0`, `1`, `2`, … modulo 8) when the request pins none, and honours an explicit mask pin. | `DT-DETERMINISTIC` | none | Because the pin is honoured, `SY-MASK` and `SY-MATRIX-EQUIV` stay clean; the defect is visible only across two calls of the identical request, which is exactly what the determinism check compares. |
| `over_capacity_acceptance` | Accepts a payload above the capacity of the pinned version and level and silently encodes a truncated payload instead of rejecting the request. | `IV-OVERCAPACITY` | none | `IV-OVERCAPACITY` builds its own over-capacity payload outside the corpus, so no corpus-driven check observes the acceptance. |

### 4.2 The nine checks that must fail

A `--adapter defective` run must fail exactly these nine check identifiers, and nothing else:

1. `RT-ROUNDTRIP`
2. `SY-VERSION-MIN`
3. `SY-EC-MATCH`
4. `SY-MATRIX-EQUIV`
5. `MD-CHARSET`
6. `RN-QUIET-ZONE`
7. `RN-MODULE-SIZE`
8. `DT-DETERMINISTIC`
9. `IV-OVERCAPACITY`

The other twelve must pass against the same target in the same run, which is the harder half to keep true:

`SY-MASK`, `MD-MODE-OPTIMAL`, `RN-CONTRAST`, `RN-SVG`, `RN-LOGO`, `DM-RESILIENCE`, `DM-SEEDED`, `IV-EMPTY`, `IV-BAD-OPTION`, `PF-ENCODE`, `PF-DECODE`, `CL-ARTIFACTS`.

Both lists are declared once, in `qrcheck/registry/expectations.py`: `SELF_CHECK_TARGETS` maps `reference` to the empty set and `defective` to the union of five per-defect sets, and the module refuses to import if that union is not the nine identifiers. `qrcheck/targets/defective.py` derives its own answer from its own `DEFECTS` table and refuses to return unless it agrees, so an edit to either declaration breaks a process instead of quietly moving the acceptance criterion. The two declarations are independent on purpose: `qrcheck list` imports the registry, and importing a concrete adapter there would pull an encoder into a process that only wants to print titles.

**The gate is set equality, not containment.** `tests/integration/test_defective_run.py::test_failed_check_set_equals_the_declared_expectation` compares the set of check identifiers that failed or errored in a real `--adapter defective` run against `SELF_CHECK_TARGETS["defective"]`. A tenth failure and a missing one both fail the gate: the first would mean a check is reporting something the injections do not explain, the second that a check cannot see the defect it exists for. `::test_the_other_twelve_checks_all_pass` holds the other half, and `::test_defects_table_declares_the_five_injections` holds the injections themselves.

### 4.3 The `reference` target: nothing may fail, and nothing may skip

`qrcheck run --adapter reference` is the clean-run gate and must produce zero `failed`, zero `error` **and** zero `skipped`, with at least one `passed` result for each of the twenty-one checks. The zero-skip requirement is deliberate: a skip would let a check such as `RN-SVG` pass CI without rasterizing anything, so CI installs the rasterizer and the gate then demands that every check actually ran. Both built-in targets declare all thirteen capabilities, so no expected failure can be turned into a skip either, and the per-check reachability gate compares the registered set, the printed set and the executed set.

### 4.4 No check may be weakened, skipped, retried or relaxed

The rule this section exists to state: no check is weakened, skipped, marked as an expected failure, retried or threshold-relaxed to make the `defective` target pass. The defective adapter and its expected failure set are deliverables in their own right, asserted by CI, and a change to either is a change to the acceptance criteria rather than a fix.

Four things hold the rule structurally rather than by convention: the expected set is validated against the registry, so deleting an expectation breaks a test; the clean-run gate forbids skips, so a check cannot pass by declining to run; each check declares the number of adapter invocations it makes and that number is asserted equal (section 6), so a retry is visible; and thresholds live only in the configuration, so a "relaxation" has to change a declared default and the criterion that reads it (`configuration.md`, `performance.md`).

## 5. Cascade containment: why the expected set stays exactly enumerable

Each of the following is a rule of the product, not a convenience, and each exists so that one injection cannot fail a check it does not target.

1. **The damage budget comes from the symbol, not from the request.** `DM-RESILIENCE` derives its occlusion budget from the error-correction level decoded out of the symbol. If it used the requested level, the `wrong_ec_level` injection would push damage past the symbol's real recovery capacity and eight damage cases would fail, and the failure set would stop being attributable. Error-level correctness is `SY-EC-MATCH`'s job; damage resilience asks only whether a symbol survives what its own level promises.
2. **The logo assertion follows the same rule.** `RN-LOGO` asserts a decode only when the covered fraction fits the budget derived from the decoded level minus the configured `occlusion_margin`. Without that rule the requested level-H logo case would become a level-Q symbol carrying the same overlay and the check would fail for a second reason.
3. **The over-capacity probe also exceeds the level below.** `IV-OVERCAPACITY` builds a payload that overflows even the error level beneath the requested one, so a target that encodes lower than asked cannot rescale it into an acceptable range and the capacity assertion keeps its single cause.
4. **`DM-SEEDED` never calls the adapter a second time.** It re-applies the recorded damage to the symbol it already holds, so a mask that changes between calls cannot leak into a reproducibility check.
5. **The mask pin is honoured.** The `non_deterministic_mask` injection honours an explicit pin, which is why `SY-MASK` and `SY-MATRIX-EQUIV` stay clean and the defect is caught by the determinism check alone.
6. **The corpus guarantees every trigger is live.** The builder guarantees at least one record in a `MD-CHARSET` class whose payload is longer than 64 bytes *and* whose 64-byte truncation still selects the same version at the run's level, so the truncation injection actually fires; at least one boundary payload whose minimal version differs between the requested level and the one below it, so `SY-VERSION-MIN` actually sees the shift; and the eight `SY-MATRIX-EQUIV` payloads are at most 64 bytes by construction, so the truncation injection can never reach them. Without those guarantees an injection would be silently inert on a small corpus, which is the failure mode this rule prevents.

When one of these rules is broken, the symptom is a defective-run gate failure listing an identifier the injections do not explain. The fix belongs in the rule, never in the check's threshold.

## 6. A check that cannot fail is a defect

Every check in this document carries two cases in the suite: a positive one, which the conforming target supplies, and a negative one, which a stub supplies. The negative case is what proves the check can fail at all.

`tests/unit/test_checks_negative.py` drives each of the twenty-one checks with a protocol-conforming stub adapter and a stubbed decode result carrying the one defect that check exists to detect — a payload differing in one byte, a wrong error level in the format information, a quiet zone of one module, a module edge off by more than the tolerance, a matrix that differs from the oracle at pinned parameters, a mask of 9, an encode that returns non-image bytes, a run that leaves a temporary entry behind. Two assertions are made, and both matter:

- **The failure is the documented one.** The result's reason must equal the reason the owning module documents for that defect, not merely `failed`. A case that failed for an unrelated reason — a stub returning bad bytes, a missing fixture — would satisfy "it failed" and say nothing about the assertion under test.
- **The control passes.** Every check also runs a round with no defect injected, which must pass, so a negative round is attributable to the injected defect rather than to the harness.

| Check | Declared `generate` invocations | Note |
|-------|--------------------------------|------|
| `DT-DETERMINISTIC` | twice per case | its defect is only visible across two calls of the identical request |
| `DM-RESILIENCE` | one per seven cases | one symbol per sampled record, shared by its seven damage variants |
| `PF-ENCODE` | `warmup` + `encode_samples` | read from the resolved configuration, never a literal |
| `DM-SEEDED`, `PF-DECODE`, `CL-ARTIFACTS` | zero | target-independent: the symbol is already held, the probe image is the framework's own, the observation is of the run |
| the other sixteen | one per case | |

The counts are asserted **equal** to the observed number, which is what makes a retry detectable: a retried case registers as the declared count plus one and fails that test, so "run it again until it passes" cannot be introduced quietly. `::test_declared_invocations_match_the_family_tables` holds these numbers against each family module's own declared expectations so the two cannot drift, and `::test_a_skipped_round_is_never_counted_as_a_pass` holds the skip rule of section 1 against a real skip.

The owning modules document three reasons a reader might expect to be otherwise, and this document follows the code: `SY-MASK`, `IV-BAD-OPTION` and `CL-ARTIFACTS` report `value_mismatch` for their injected defect (and `CL-ARTIFACTS`'s skip reason is `unsupported_platform`), while `RN-SVG`'s round carries a version with no payload and so reports `value_mismatch` rather than `sampling_failed`. The reasons stated in each check's row above are the ones the code emits.

## 7. States a result can reach, and the gate that asserts each

Each row is a state a reader will actually meet, the result it produces, and the named test that holds the behaviour in place. `architecture.md` owns *why* the consensus layer classifies as it does; this table owns what the reader sees.

| State | What the result looks like | Gate that asserts it |
|-------|---------------------------|----------------------|
| A decoder returns nothing | That decoder's `per_decoder` entry has `status` `undetected` with its own "no symbol" text in `error_text`. An `undetected` decoder is **never** a disagreement; when the primary decoder is the silent one, the round-trip check fails with `value_mismatch`, because the authority produced nothing to compare. The result's `secondary_comparable` is `null`, which is what tells this state apart from a `primary_only` result whose secondary answer existed but could not be compared (`secondary_comparable` `false`, section 8.2). | `tests/unit/test_decoder_consensus.py::test_silence_in_both_decoders_is_undetected_and_never_a_disagreement` |
| The two decoders disagree | `failed` with reason `decoder_disagreement`; `per_decoder` holds both decoders' payloads and metadata blocks verbatim, and both are kept in `decoded.json`. The framework does not pick a winner. Every check that consumes a decoded payload reports it — `RT-ROUNDTRIP`, `MD-CHARSET` and the four `SY-*` checks — so one symbol cannot pass one check and fail another on the consensus classification alone. | `tests/unit/test_decoder_consensus.py::test_payload_disagreement_fails_roundtrip_with_decoder_disagreement` |
| The format information disagrees with the decoder's level | `failed` with reason `format_info_disagreement`, `ec_level_decoder` and `ec_level_format` both recorded beside `request_ec_level`. | `tests/unit/test_decoder_consensus.py::test_sym_ec_match_fails_with_format_info_disagreement_and_keeps_both_values` |
| The target returns an image that will not decode | The case fails: `value_mismatch` for the round-trip check, whose `per_decoder` entry keeps the decoder's own "no symbol" text verbatim, and `value_mismatch` with `problems` naming `raster_does_not_decode` for `RN-SVG`. The artefact is kept as `symbol.png`, so the symbol can be looked at rather than only described. | `tests/unit/test_checks_negative.py` (the `undecodable-but-measured` round), `tests/unit/test_decoder_consensus.py::test_silence_in_both_decoders_is_undetected_and_never_a_disagreement` |
| The target returns non-image bytes | The adapter raises `TargetContractError`; the case fails with `contract_error` and `decoded.json` carries `error_text` plus `hex_prefix`, the first 256 bytes of the unusable body as 512 hexadecimal characters (fewer when the body is shorter, empty when there is no body). A check that has to open the artefact itself — the damage family reads the pixels it damages — reports the same `contract_error` result for the same reason and never lets the image library's exception abort the run. | `tests/unit/test_targets_errors.py::test_library_adapter_non_image_bytes_record_the_full_hex_prefix` |
| The target lacks a capability the check requires | `skipped` with reason `unsupported_capability`, resolved before execution: `counts.skipped` increments and `counts.passed` is unchanged, so the run cannot look green. | `tests/unit/test_registry_contract.py::test_a_capability_shortfall_resolves_to_a_skip_and_never_to_a_pass` |
| No module grid can be recovered | `failed` with reason `sampling_failed` — a stated inability to observe, never a wrong verdict about the target. The recording carries the module-count estimate: the estimated count, the pitch, the finder run in pixels, the snapped legal dimension and the timing-pattern transition counts that refused it. | `tests/unit/test_matrix_sampling.py::test_both_paths_refusing_yields_sampling_failed`, `::test_version_min_reports_sampling_failed_when_no_grid_is_recovered`, `::test_quiet_zone_reports_sampling_failed_when_no_pitch_can_be_measured` |
| The target is unreachable or times out | Exactly one `error` result per affected check, filed under `synthetic:target` with reason `target_unavailable` or `target_timeout` — not one per corpus record. Checks that need nothing from the target still run and still pass, and the process exits with the internal-error code. | `tests/unit/test_runner_target_loss.py::test_an_unreachable_target_files_one_error_per_affected_check`, `::test_a_timed_out_target_files_one_error_per_affected_check_naming_the_timeout`, `::test_the_run_continues_to_the_checks_that_need_no_target` |
| A check or the framework raises | `error` with reason `internal_error`; the traceback travels in the result with its absolute paths folded into a portable form, later checks still run, and the run exits with the internal-error code. | `tests/unit/test_exit_codes.py::test_exit_three_when_a_check_raises_internally`, `tests/unit/test_runner_target_loss.py::test_a_run_whose_check_errored_exits_with_the_internal_code` |

## 8. Reading the evidence

### 8.1 Layout and policy

```
<report_dir>/evidence/<check-id>/<failure-id>/
```

`<check-id>` is the identifier exactly as this document writes it, and `<failure-id>` is the derivation of section 1.3 — sanitized, so `corpus:` appears as `corpus_` and `@` as `_`. Evidence is written for a `failed`, an `error` and a `skipped`-with-a-reason case; a `passed` case records `evidence_dir: null` and writes nothing, which is what keeps a full run's file count proportional to what went wrong rather than to what was asserted.

The artefact set is closed. A check writes only these names:

| Artefact | Written by | Content |
|----------|-----------|---------|
| `request.json` | every case that writes evidence | The request in serializable form, plus `check_id`, `case_id` and `seed`. `out_of_contract: true` marks a deliberate `IV-*` probe, and a case's extra resolved facts — the damage kind and the thresholds actually applied, the quiet-zone request, the contrast request, the oracle digest — are merged in beside them. |
| `symbol.png` / `symbol.svg` | every image-producing check | The artefact the target returned, byte for byte as received. |
| `decoded.json` | every case that writes evidence | The decode projection, the target's own claim about the symbol it returned and the check's own measurements (8.2). |
| `symbol-repeat.png` | `DT-DETERMINISTIC` only | The second call's artefact for the identical request — what makes a mask that changes between calls observable. |
| `damage-<kind>-<seed>.png` | `DM-RESILIENCE`, `DM-SEEDED` | One per variant, with `<kind>` from the declared order and `<seed>` the decimal derived seed. |
| `logo.png` | `RN-LOGO` only | The generated overlay bitmap that was composited, so the covered area can be re-measured. |

Both JSON files are written canonically: sorted keys, two-space indent, UTF-8 with the characters as themselves, and a single trailing newline. Neither carries a timestamp or a duration, so the same case at the same seed produces byte-identical evidence — which is what makes `DM-SEEDED` a meaningful check and what `tests/integration/test_damage_reproducibility.py` exercises across the whole damage matrix.

### 8.2 What `decoded.json` always carries, and what each check adds

Every `decoded.json` for a case that decoded a symbol has the same three parts — the canonical decode projection, the target's own claim about the symbol it returned, and the measurements the check's own assertion used. The projection is fixed, so these fields are present whatever the check:

`per_decoder` (each decoder's `decoder`, `status`, `polarity`, `payload_sha256`, `reported_ec`, `reported_version`, `error_text`), `unavailable_decoders` (the reason a decoder the run did not consult could not be, `{}` when both were consulted), `polarity_applied`, `agreement`, `symbology_identifier`, `content_type`, `ec_level_decoder`, `ec_level_format`, `version`, `mask`, `matrix_sha256`, `secondary_comparable`, and `sampling` — `null` where no grid was sampled, otherwise the module-count estimate with its pitch, finder run, snap and timing-pattern transitions. `secondary_comparable` is `true` when both decoders answered and their payloads could be compared, `false` when they answered with differing payloads the text-only secondary decoder cannot be held to (so `agreement` stays `primary_only` by the refinement `architecture.md` records), and `null` when no comparison was attempted — the secondary decoder did not run, found nothing or errored, or the primary produced no payload. A case that only observed a rejection records the rejection text in place of the payload fields, and `CL-ARTIFACTS` records its own run-level observation instead of a decode.

The claim is the part of the file that is neither projection nor measurement: `declared_version`, `declared_ec`, `declared_mask` and `declared_mode` are what the target said about the symbol it returned, taken from its `GeneratedSymbol` as evidence only and never asserted, `null` for a field the target did not state, and absent altogether from a case whose target returned no symbol — a rejection, a contract error and a pre-execution skip claim nothing.

On top of that projection and that claim each check records the measurements its assertion used:

| Check | Fields to open first |
|-------|---------------------|
| `RT-ROUNDTRIP`, `MD-CHARSET` | `payload_sha256`, `payload_len`, `decoded_len`, `first_divergence_offset` |
| `MD-MODE-OPTIMAL` | `expected_mode`, `max_version`, `payload_len_in_mode`, decoded `version` |
| `SY-VERSION-MIN` | `expected_version`, `oracle_version`, `module_count_for_expected_version`, `sampled_version` |
| `SY-EC-MATCH` | `ec_level_decoder`, `ec_level_format`, `request_ec_level` — all three side by side |
| `SY-MASK` | `mask`, `mask_pin` |
| `SY-MATRIX-EQUIV` | `oracle_matrix_sha256`, `sampled_matrix_sha256`, `differing_modules`, `first_difference`, `expected_module_count`, `sampled_module_count` |
| `RN-QUIET-ZONE` | `quiet_zone_measured` (`top`, `right`, `bottom`, `left`), `quiet_zone_requested`, `module_px_measured` |
| `RN-MODULE-SIZE` | `module_px_measured`, `module_size_px_requested`, `module_px_deviation`, `modules_across_canvas`, `implied_version` |
| `RN-CONTRAST` | `contrast_ratio_measured`, `contrast_floor`, `palette_measured`, `palette_requested`, `polarity_requested`, `inverted_request` |
| `RN-SVG` | `rasterizer` (`configured`, `name`, `available`, `attempted`), `svg_canvas_side`, `svg_module_count`, `svg_module_pitch`, `payload_matches_request`, `problems` |
| `RN-LOGO` | `decoded_ec_level`, `logo_area_fraction_measured`, `logo_area_fraction_requested`, `recovery_budget`, `assertion_applied`, `logo_seed` |
| `DM-RESILIENCE` | `pristine` (level pair, version, matrix digest) and `damage[]` with each variant's `kind`, `seed`, `agreement`, `decoded_len` and `per_decoder` |
| `DM-SEEDED` | `regenerated_sha256` against `recorded_sha256`, `first_difference`, `applied` |
| `IV-EMPTY` | `outcome` (`declared_rejection`, `documented_success`, `incomplete_artefact`) |
| `IV-OVERCAPACITY` | `requested_len` against `capacity`, `decoded_len`, `first_divergence_offset` |
| `IV-BAD-OPTION` | `offending_field`, `value`, and for the contrast probe `contrast_ratio` and `contrast_floor` |
| `DT-DETERMINISTIC` | `comparison` (`image_bytes`, `matrix_sha256`, `mask`), `divergent`, `unmeasured`, plus each call's snapshot |
| `PF-ENCODE` | `metrics.encode_p95_ms` and `metrics.symbols_per_minute` — both **measured**, so both are masked by every determinism comparison — then `metrics.samples`, `metrics.warmup`, which are configuration-derived and stay compared |
| `PF-DECODE` | `metrics.decode_p95_ms` — **measured**, so it is masked by every determinism comparison — then `metrics.image_px`, `metrics.samples`, `metrics.warmup`, which are configuration-derived and stay compared |
| `CL-ARTIFACTS` | `temp_entries_before`, `temp_entries_after`, `leaked`, `fd_directory`, `open_images`, `open_under_report_dir`, `handle_half`, `violations` |

`request.json` for the `IV-*` probes carries `out_of_contract: true`. That flag is the only thing distinguishing an intentionally illegal request from one the framework considered valid, so it is the first field to check when an `IV-*` result looks wrong: a probe that was accidentally built through the validating builder would be rejected before it ever reached the target.

## 9. What this document does not cover

- The performance floors, how a percentile is computed and which environment they are defined for — `performance.md`.
- The type, default, validation rule and diagnostic of every configuration key named above — `configuration.md`.
- The pipeline, the layering, and why a decoder's silence is not a disagreement — `architecture.md`.
- The pins, the lock files and the rasterizer binaries a skip may be telling you to install — `dependencies.md`.
- How to declare a `library`, `cli` or `http` target and what each of its keys maps to — `targets.md`.
