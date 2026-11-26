# qrcheck configuration

Every key of `qrcheck.toml`: its type, its default, the rule it must satisfy, and the diagnostic that fires when it does not. Read this when writing or repairing a configuration. What each check does with a value belongs elsewhere — `checks.md` for the assertions, `performance.md` for the meaning and measurement of the performance floors, `dependencies.md` for the rasterizer binaries and the pins, `targets.md` for the field-by-field walkthrough of declaring a `library`, `cli` or `http` target, `architecture.md` for why the layering is shaped this way.

The single owner of this contract in code is `qrcheck/config/schema.py`: one declarative table of key, type, default and rule, read by `qrcheck/config/load.py` at start-up. A key this document does not list does not exist, and writing one is an error rather than a silently ignored line.

## 1. Where the configuration lives

`qrcheck.toml` at the repository root is the committed default — the document a run uses when `--config` is not given. `qrcheck --config PATH` selects another file; `PATH` is taken as spelled, relative to the working directory, with no `~` expansion and no environment interpolation.

Everything the product can be told lives in that one document. Nothing is read from an environment variable or an environment file, there is no secret, no profile to inherit, no include to merge and no per-environment directory: a run is a function of the document plus the command-line options of section 9, and of nothing else. A second `qrcheck.toml` is the only way to have a second configuration.

## 2. How a file resolves, loads and is validated

One load, six steps, all before any check runs.

1. Read the file's bytes. An unreadable path — missing, a directory, denied — is a configuration error, never an environment error.
2. Decode as UTF-8.
3. Parse with the standard library's `tomllib`. The parser's own `at line N column M` position is kept verbatim, because it is what turns the diagnostic into an instruction.
4. Validate the whole document against the schema table.
5. Fill in every declared default the document omits.
6. Resolve the two path keys, create `report_dir` and prove it writable, then build the frozen records the rest of the product reads.

### Paths resolve against the configuration file's directory

`report_dir` and `corpus_path` are resolved against the directory holding the configuration file, not the working directory the command was launched from, so `qrcheck --config /anywhere/qrcheck.toml` behaves the same for every user and from every directory. An absolute value in the document is honoured as written. Neither `~` nor `$VAR` is expanded, and a path is not checked for existence at load time.

### Validation runs before any check, so a bad file never produces a report

A rejected key exits 2 with no report written — not a truncated report, not an empty one. The diagnostic is `<dotted path>: <message>`, and each command prefixes it:

```
qrcheck run: configuration error: targets.mygen.returns: invalid value 'bytes'; allowed: ['base64', 'bytes_path', 'file_path', 'raw_bytes']
```

Every diagnostic names the key by its **dotted path** — `targets.mygen.returns`, `checks.damage.jpeg_quality` — so the message is the edit to make. An unknown key reports itself twice over, once as the path that has to change and once inside the message: `targets.mygen.endpoint: unknown key 'targets.mygen.endpoint'`. A fault that belongs to the whole file rather than to one key carries no path and reads as one sentence, naming the file by its resolved absolute path — a relative `--config PATH` is resolved against the working directory first, which is what puts a full path in the message: `cannot read configuration file: <resolved path>`, `configuration file is not valid UTF-8: <resolved path>`, `invalid TOML in <resolved path>: Expected '=' after a key in a key/value pair (at line 12, column 8)`.

The first diagnostic is stable, which matters when a document has more than one mistake: unknown keys are reported over sorted names, required keys in the table's declared order, and the sections are walked in a fixed sequence — unknown top-level keys, then top-level values, `[render]`, `[checks]`, `[targets]`, and last the cross-key rule below. A given document therefore always reports the same fault first, on every machine and every run.

### The one cross-key rule

`default_adapter` must name a declared `[targets.<name>]` table. Because the key's fallback is `reference`, a document that declares no target at all cannot satisfy it and fails with `default_adapter: unknown adapter 'reference'; declared: []`. A usable configuration declares at least one target; the committed file declares both built-in ones.

### What the loader deliberately does not check

- **Whether `corpus_path` exists.** That condition holds only when a corpus-driven check is selected, which the loader cannot know, so the run raises it before any check as a run-prerequisite error — exit 2, naming the resolved path.
- **Whether a declared capability set satisfies the checks.** It depends on the run-time `--checks` selection; see section 8.
- **Whether a `library` target's module and attribute exist.** Syntax only here; `qrcheck doctor` and, at run time, `TargetUnavailable` answer existence, because the same document is valid on a machine where the target is installed.

`report_dir` is the one filesystem fault that is not a configuration error. It is created if absent and then proved writable the only way it can be — by writing a probe file into it and removing it again — so a read-only directory, a full filesystem or a path whose parent is a regular file each surfaces as the failure the real write would have produced. That is exit 3, never exit 2 (section 8).

## 3. Top-level keys

Five scalar keys, all optional in the document because each has a declared default.

| Key | Type | Default | Validation | Diagnostic on violation |
|-----|------|---------|------------|-------------------------|
| `default_adapter` | string | `"reference"` | non-empty; must name a declared `[targets.<name>]` table | exit 2 — `default_adapter: unknown adapter '<x>'; declared: <sorted list>` |
| `report_dir` | path | `"reports"` | non-empty, no NUL byte; resolved against the configuration file's directory; created if absent and proved writable before a run | **exit 3**, not exit 2 — `report directory cannot be written: <dir>: <detail>` |
| `seed` | integer | `20240101` | `>= 0` | exit 2 — `seed: seed must be a non-negative integer` |
| `corpus_path` | path | `"corpus/default.jsonl"` | non-empty, no NUL byte; resolved like `report_dir`; must exist when a corpus-driven check is selected | exit 2, naming the resolved path |
| `loc_limit` | integer | `60000` | `>= 1` | exit 2 — `loc_limit: must be at least 1, got <n>` |

The `default_adapter` and `seed` messages above are literal and must stay so. Every other row names the offending key by its dotted path and states what was expected against what arrived; a boolean is rejected wherever an integer is accepted, because TOML `true` parses to an `int` in Python and `warmup = true` would otherwise pass as `1`.

`seed` is the run seed every derived seed comes from — corpus payloads, occluder placement, noise, the logo bitmap — so a run is reproducible from this value together with the adapter and the EC level recorded in `summary.json`.

## 4. The keys every target shares

A target is one `[targets.<name>]` table. The name is the identifier `--adapter` and `default_adapter` use, and it must be a non-empty string. `type` selects the per-type keys, so a key belonging to another kind — `command` on a `library` target — is an **unknown key** rather than a silently inert one.

| Key | Type | Default | Validation | Diagnostic on violation |
|-----|------|---------|------------|-------------------------|
| `type` | string | required | one of `library`, `cli`, `http`, `reference`, `defective` | exit 2 — `targets.<name>.type: invalid value '<v>'; allowed: ['cli', 'defective', 'http', 'library', 'reference']`, or `targets.<name>.type: missing required key 'type'` |
| `timeout_s` | number | `30.0` | `> 0` | exit 2 — `targets.<name>.timeout_s: must be greater than 0, got <n>` |
| `contrast_floor` | number | `7.0` | `> 1.0` | exit 2 — `targets.<name>.contrast_floor: must be greater than 1, got <n>` |
| `capabilities` | array of strings | `[]` | every member one of the thirteen names below | exit 2 — `targets.<name>.capabilities: unknown capability '<m>'; allowed: <sorted list>` |

`timeout_s` bounds one operation against the target, not the run. Exceeding it raises `TargetTimeout`, which the check reports as an error naming the configured timeout — the run continues with the next check.

`contrast_floor` is the target's own declaration of the luminance ratio it will render, and it has two readers: `RN-CONTRAST` asserts a normal request sits at or above it, and `IV-BAD-OPTION`'s fourth probe builds a colour pair measured below it to prove the target rejects that request rather than rendering it. Its default is exactly the ratio `RN-CONTRAST` asserts, so a target that declares nothing is still held to that floor.

`capabilities` is what the target says it can do. The set is closed at thirteen members:

| Capability | What a target declaring it promises |
|------------|-------------------------------------|
| `png` | produce a raster PNG artefact |
| `svg` | produce a vector SVG artefact |
| `ec_level` | honour the requested error-correction level |
| `module_size` | honour a requested module edge length |
| `quiet_zone` | honour a requested quiet-zone width |
| `color` | honour requested foreground and background colours |
| `logo_overlay` | composite a supplied overlay bitmap onto the symbol |
| `version_pin` | honour a pinned version |
| `mask_pin` | honour a pinned mask |
| `metadata_version` | report the symbol's version as metadata |
| `metadata_ec` | report the symbol's error-correction level as metadata |
| `metadata_mask` | report the symbol's mask as metadata |
| `metadata_mode` | report the symbol's encoding mode as metadata |

An unknown member is a configuration error, because it is wrong whoever selects what. A **shortfall** — a valid set that does not cover a selected check — is not: it is a `skipped` result. Both rules, and the reason they differ, are in section 8.

A `reference` or `defective` target accepts these four keys and no others. Their per-type tables are empty by design, so a configurable input on a built-in target is an unknown key rather than a key with no effect.

## 5. Per-type keys

### 5.1 `type = "library"` — an in-process import and call

| Key | Type | Default | Validation | Diagnostic on violation |
|-----|------|---------|------------|-------------------------|
| `callable` | string | required | exactly one `:`, both parts non-empty, and every dotted part a Python identifier | exit 2 — `targets.<name>.callable: expected exactly one ':' as 'module:function', got '<v>'`, or `… is not a valid Python identifier in '<v>'` |
| `kwargs_map` | table of strings | required | every key and every value a non-empty string, and every key one of the `GenerateRequest` field names — that vocabulary is the adapter's, so it is refused when the adapter is built, still as exit 2 naming the key | exit 2 — `targets.<name>.kwargs_map: every key must be a non-empty string, got <v>`, or `targets.<name>.kwargs_map: kwargs_map names unknown framework field(s) '<f>'; expected any of payload, ec_level, …` |
| `returns` | string | required | one of `bytes_path`, `file_path`, `base64`, `raw_bytes` | exit 2 — `targets.<name>.returns: invalid value '<v>'; allowed: ['base64', 'bytes_path', 'file_path', 'raw_bytes']` |
| `metadata_map` | table of strings | `{}` (optional) | keys one of `version`, `ec`, `mask`, `mode`; values non-empty strings | exit 2 — `targets.<name>.metadata_map: names unknown framework field(s) '<f>'; expected any of version, ec, mask, mode` |

`callable` is a `module:function` path resolved at run time. `kwargs_map` maps a framework request field to the keyword argument the callable takes for it, so a generator whose parameter is named `data` and whose error level is called `level` is wired without touching the target's code. `returns` says how the returned value carries the artefact: a path to bytes, a path to a written file, base64 text, or the bytes themselves. `metadata_map` is evidence only — what the target claims about the symbol, recorded beside what the decoders found and never asserted.

The vendor's shape lives entirely in this table: the adapter reads the mappings and hard-codes no field name, no keyword argument and no return form.

### 5.2 `type = "cli"` — an argv-only subprocess

| Key | Type | Default | Validation | Diagnostic on violation |
|-----|------|---------|------------|-------------------------|
| `command` | array of strings | required | non-empty; every element a non-empty string | exit 2 — `targets.<name>.command: must list at least one argument`, or `argument <i> must be a string, got <kind>` |
| `payload_arg` | string | required | contains `{}`, or starts with `-` | exit 2 — `targets.<name>.payload_arg: must contain '{}' or start with '-', got '<v>'` |
| `output_arg` | string | required | contains `{}`, or starts with `-` | exit 2 — `targets.<name>.output_arg: must contain '{}' or start with '-', got '<v>'` |
| `returns` | string | required | one of `bytes_path`, `file_path`, `base64`, `raw_bytes` | exit 2 — `targets.<name>.returns: invalid value '<v>'; allowed: ['base64', 'bytes_path', 'file_path', 'raw_bytes']` |
| `metadata_map` | table of strings | `{}` (optional) | keys one of `version`, `ec`, `mask`, `mode`; values non-empty strings | exit 2 — as for `library` |

Execution is argv-only: the command runs with `shell=False`, no shell splits or expands it, and each invocation is bounded by `timeout_s`. That is why `command` is an array and not a shell string — a string would reintroduce exactly the quoting and injection surface the array removes — and why each argument template is either a `{}` placeholder for the value or a flag beginning with `-` that receives it.

The vendor's shape lives entirely in this table: the adapter builds an argv list from the configured templates and knows no flag name of its own.

### 5.3 `type = "http"` — a loopback request

| Key | Type | Default | Validation | Diagnostic on violation |
|-----|------|---------|------------|-------------------------|
| `url` | string | required | scheme `http` or `https`, host one of `127.0.0.1`, `localhost`, `::1`, and no `user:password@` prefix | exit 2 — `targets.<name>.url: url host '<h>' is not a loopback host; allowed: ['127.0.0.1', '::1', 'localhost']` |
| `method` | string | required | `GET` or `POST` | exit 2 — `targets.<name>.method: invalid value '<v>'; allowed: ['GET', 'POST']` |
| `request_template` | table, or a string holding a JSON object | required | every string leaf's `{…}` field name drawn from the eight placeholders below; values limited to what JSON carries — no TOML dates or times | exit 2 at the leaf — `targets.<name>.request_template.<leaf>: unknown placeholder '<f>' in request_template; allowed: ['bg', 'ec_level', 'fg', 'format', 'margin', 'payload_b64', 'payload_utf8', 'size']` |
| `image_field` | string | required | `_body`, or a dotted path of non-empty, whitespace-free segments | exit 2 — `targets.<name>.image_field: expected '_body' or a dotted path of non-empty segments, got '<v>'` |
| `image_encoding` | string | required | `base64` or `raw` | exit 2 — `targets.<name>.image_encoding: invalid value '<v>'; allowed: ['base64', 'raw']` |
| `metadata_map` | table of strings | `{}` (optional) | keys one of `version`, `ec`, `mask`, `mode`; values dotted paths | exit 2 — as for `library` |

The eight placeholders a `request_template` may interpolate, and nothing else:

| Placeholder | Interpolated with |
|-------------|-------------------|
| `{payload_b64}` | the payload, base64-encoded |
| `{payload_utf8}` | the payload as UTF-8 text |
| `{ec_level}` | the requested error-correction level, `L`/`M`/`Q`/`H` |
| `{format}` | the requested output format, `png` or `svg` |
| `{size}` | the requested module edge in pixels |
| `{margin}` | the requested quiet zone in modules |
| `{fg}` | the requested foreground colour |
| `{bg}` | the requested background colour |

A placeholder may carry a conversion or a format spec — `{size:d}`, `{payload_utf8!s}` — and only the field name is checked; escaped braces (`{{`, `}}`) are literal text. A `request_template` may be written as a TOML table or as a string holding that object's JSON, and both forms resolve to the same mapping, because the body is sent as JSON either way.

**The host rule is the offline constraint's enforcement point.** `qrcheck` opens exactly one kind of socket — to a target it was told about — and it refuses to be aimed anywhere but the loopback interface: `http://example.com/generate` and `http://192.168.0.10:8000/generate` are exit 2 naming the host, while `http://[::1]:8000/` is accepted. The allow-list is declared once, as `models.LOOPBACK_HOSTS`, and read by both the schema and the HTTP adapter, so the boundary cannot widen on one side while staying closed on the other. A `user:password@` prefix is refused in the same spirit: there is nothing to authenticate to on loopback, and this configuration holds no credentials.

A `url` rejected on scheme or on credentials says so in its own words rather than reporting a host: `targets.<name>.url: url scheme 'ftp' is not http or https; allowed: ['http', 'https']`, and `targets.<name>.url: url must not carry a username or password; configuration holds no credentials`.

The vendor's shape lives entirely in this table: endpoint, method, request body, where the image sits in the response and how it is encoded are all configurable, and the adapter contains no field name, endpoint or response shape of its own.

## 6. Three tables beyond the request's key list

The key list above is closed, and three tables are added to it. Each exists for something the product's own specification requires to be configurable while giving it no home: a rasterizer to name, performance floors that must live in configuration, and damage thresholds whose change point must be the configuration file. Every other key is unchanged, and an unknown key anywhere still exits 2.

Each threshold has exactly one home — the committed `qrcheck.toml` entry, with the `models.DEFAULT_*` constant as the fallback for a file that omits it. No check module, test or document carries a numeric literal for a threshold, so retuning one means changing the value here plus the acceptance criterion that reads it, and nothing else.

### 6.1 `[render]` — how SVG output is rasterized

| Key | Type | Default | Validation | Diagnostic on violation |
|-----|------|---------|------------|-------------------------|
| `svg_rasterizer` | string | `"auto"` | one of `auto`, `cairosvg`, `rsvg-convert`, `resvg`, `magick`, `none` | exit 2 — `render.svg_rasterizer: invalid value '<v>'; allowed: ['auto', 'cairosvg', 'magick', 'none', 'resvg', 'rsvg-convert']` |

`RN-SVG` asserts that an SVG artefact rasterizes to a decodable matrix, so it needs a rasterizer, and this key names which one. `auto` discovers in a fixed order: an importable `cairosvg` module first, then `rsvg-convert`, `resvg`, `magick` and `convert` on `PATH`. The order is a contract rather than a preference, because `doctor` and `RN-SVG` both report the sequence they attempted.

`none` disables rasterizing. `RN-SVG` then reports `skipped` with reason `missing_dependency` and records the order it tried — a skip, never a pass. The same happens when `auto` finds nothing. `dependencies.md` covers the binaries and what each absence costs; `qrcheck/checks/render.py` is the single discovery and invocation implementation.

### 6.2 `[checks.performance]` — the latency and throughput floors

| Key | Type | Default | Validation | Diagnostic on violation |
|-----|------|---------|------------|-------------------------|
| `encode_p95_ms` | number | `250.0` | `> 0` | exit 2 — `checks.performance.encode_p95_ms: must be greater than 0, got <n>` |
| `decode_p95_ms` | number | `400.0` | `> 0` | exit 2 — `checks.performance.decode_p95_ms: must be greater than 0, got <n>` |
| `min_symbols_per_minute` | number | `120.0` | `> 0` | exit 2 — `checks.performance.min_symbols_per_minute: must be greater than 0, got <n>` |
| `encode_samples` | integer | `100` | `>= 10` | exit 2 — `checks.performance.encode_samples: must be at least 10, got <n>` |
| `decode_samples` | integer | `200` | `>= 10` | exit 2 — `checks.performance.decode_samples: must be at least 10, got <n>` |
| `warmup` | integer | `5` | `>= 0` | exit 2 — `checks.performance.warmup: must be at least 0, got <n>` |
| `decode_image_px` | integer | `500` | `>= 64` | exit 2 — `checks.performance.decode_image_px: must be at least 64, got <n>` |

The two probe checks read this table and nothing else: `PF-ENCODE` asserts the encode p95 and the throughput floor, `PF-DECODE` asserts the decode p95 on one fixed PNG of `decode_image_px` × `decode_image_px` pixels, and `warmup` counts the untimed operations that precede both measurements so a reported p95 reflects a warmed process rather than interpreter start-up. `PF-ENCODE` copies the configured threshold into each result's `expected`, so a report states the floor it was judged against. The sample-count floors exist so a percentile is measured from enough observations to mean anything. `performance.md` owns the budget environment, the measurement method and what the floors mean.

### 6.3 `[checks.damage]` — the damage thresholds

| Key | Type | Default | Validation | Diagnostic on violation |
|-----|------|---------|------------|-------------------------|
| `occlusion_margin` | number | `0.05` | `>= 0` and `< 0.10` | exit 2 — `checks.damage.occlusion_margin: must be less than 0.1, got <n>` |
| `blur_radius_per_module` | number | `0.6` | `> 0` | exit 2 — `checks.damage.blur_radius_per_module: must be greater than 0, got <n>` |
| `jpeg_quality` | integer | `40` | `1`–`95` | exit 2 — `checks.damage.jpeg_quality: must be at most 95, got <n>` |
| `noise_sigma` | number | `12.0` | `>= 0` | exit 2 — `checks.damage.noise_sigma: must be at least 0, got <n>` |
| `rotation_deg` | number | `15.0` | `> 0` and `<= 45` | exit 2 — `checks.damage.rotation_deg: must be at most 45, got <n>` |
| `perspective_deg` | number | `10.0` | `> 0` and `<= 30` | exit 2 — `checks.damage.perspective_deg: must be at most 30, got <n>` |
| `downscale_px_per_module` | number | `3.0` | `> 0` | exit 2 — `checks.damage.downscale_px_per_module: must be greater than 0, got <n>` |
| `logo_area_fraction` | number | `0.15` | `> 0` and `<= 0.25` | exit 2 — `checks.damage.logo_area_fraction: must be at most 0.25, got <n>` |

`DM-RESILIENCE` applies one variant per key — occluders, Gaussian blur, JPEG re-encode, additive noise, rotation, perspective warp, downscale — to one symbol per sampled corpus record, and `DM-SEEDED` re-applies each from the recorded seed. `logo_area_fraction` is read by `RN-LOGO`, the overlay's area as a fraction of the symbol's.

`occlusion_margin` is the one key whose meaning needs a sentence here, because it is a **safety margin measured against error-correction capacity**, not a shape. The occlusion budget is the documented recovery capacity of the EC level decoded from the symbol — `L` `0.07`, `M` `0.15`, `Q` `0.25`, `H` `0.30` of codewords — with this margin subtracted, because the documented percentage is approximate and real tolerance can be lower. Occluders are placed only in the data area: never over a finder pattern or its separator, either timing pattern, an alignment pattern, either copy of the format information, a version-information block, or the fixed dark module. Covering a function pattern makes a symbol undetectable however much error-correction capacity remains, which would fail a conforming target for a reason that has nothing to do with the capacity being tested. The upper bound below `0.10` is what keeps the margin from consuming the whole `L`-level capacity.

## 7. The validation this file does not own

Only request *shape* is validated here. Whether a request is one the framework is allowed to send is decided by `build_request()` in `qrcheck/targets/base.py` — the single constructor every ordinary case goes through, and therefore the single definition of "the framework considers this request valid". It refuses:

- a payload that is not bytes-like (text is refused rather than encoded, so the caller's encoding choice stays visible);
- an `ec_level` outside `L`, `M`, `Q`, `H`, or an `output_format` outside `png`, `svg`;
- a pinned `version` outside `1`–`40`, or a pinned `mask` outside `0`–`7`;
- a `module_size_px` below `1`, or a negative `quiet_zone_modules`;
- a colour channel outside `0`–`255`, or anything other than three of them;
- a payload that does not fit the requested version — `40` when no version is pinned — at the requested level, in the payload's own mode.

`GenerateRequest` itself validates nothing. The gate is the builder, not the type, which is what makes the invalid-input checks possible: `IV-EMPTY`, `IV-OVERCAPACITY` and the four `IV-BAD-OPTION` probes exist to prove a target **rejects** a request that breaks exactly those rules, so they cannot travel through the builder. They construct the same frozen record through `construct_out_of_contract()` in `qrcheck/checks/support.py`, which performs no validation at all, and pass `out_of_contract=True` when recording evidence so that `request.json` carries `out_of_contract: true` and no reader mistakes an intentionally illegal request for a framework defect. Calling that constructor for an ordinary case is a defect in the framework.

The four `IV-BAD-OPTION` probes are an `ec_level` of `"X"`, a `module_size_px` of `0`, a `quiet_zone_modules` of `-1`, and a colour pair measured below the target's declared `contrast_floor`. A `ValueError` escaping `build_request()` is a framework bug rather than a target defect: the runner turns it into an `error` result with its traceback, and the run exits 3.

## 8. Exit 2, exit 3, and the one documented skip

Three outcomes, and which one a condition gets is part of the contract.

| Group | Conditions | Outcome |
|-------|-----------|---------|
| **Exit 2** — decided before any check runs, no report written, offending key or selection named | an unknown key at any level — top level, `[render]`, `[checks.<child>]`, `[targets.<name>]`, or a per-type key on a target whose `type` does not take it; a missing required key — a target's `type`, its `callable`/`kwargs_map`/`returns`, its `command`/`payload_arg`/`output_arg`, or its `url`/`method`/`request_template`/`image_field`/`image_encoding`; an enum value outside its set — `type`, `returns`, `method`, `image_encoding`, `svg_rasterizer`; a syntactically invalid `callable`; an empty `command` or a malformed argument template; a non-loopback `url`, a scheme other than `http`/`https`, or a `user:password@` URL; a `timeout_s` at or below `0` or a `contrast_floor` at or below `1.0`; an unknown `capabilities` member; a negative `seed`, a `loc_limit` below `1`, or any performance or damage value outside its range; an unreadable file, non-UTF-8 bytes or invalid TOML; a `default_adapter` that names no declared target; a `corpus_path` that does not exist once a corpus-driven check is selected; a `--checks` selector that matches no registered check, echoed verbatim; a `--class` filter that leaves no record, named; a `kwargs_map` naming a field `GenerateRequest` does not have, or mapping two of its fields onto one callable kwarg, a `metadata_map` beside a stdout-based `returns`, which would read one stream twice, or an `http` `request_template` with no payload placeholder — the per-target vocabularies the schema checks for shape only, refused when the adapter is built and named by their own dotted key; a target the adapter constructor finds unavailable | exit 2, stderr names the key, filter or dependency, no report |
| **Exit 3** — an environment or framework fault | `report_dir` cannot be created or written; any internal error after execution began | exit 3; the diagnostic names the directory, and a run whose report cannot be written streams every result it holds to stdout as one JSON object per line, so a consumer is never left with nothing. A `report_dir` found unwritable before the run starts holds no result to stream, so the diagnostic alone is written and says so |
| **A skip** — a result, counted separately, never a pass | a selected check whose requirements the target's declared `capabilities` cannot satisfy — reason `unsupported_capability`, listing the capability names to add; a declared optional runtime dependency that is absent, such as the SVG rasterizer — reason `missing_dependency`; a platform without the descriptor directory an observation needs — reason `unsupported_platform` | `skipped`, tallied in `counts.skipped` with `counts.passed` unchanged |

Precedence: exit 2 is settled before execution and so never competes with a result; during and after a run, `3` outranks `1`, which outranks `0`.

**The documented exception.** The capability shortfall is the one condition that does not produce exit 2, and the reason is that it cannot be decided from the document alone. Whether a declared capability set is sufficient depends on the run-time `--checks` selection, so the same, valid configuration file would be an error for one selection and correct for the next. `qrcheck/registry/resolve.py` therefore intersects each selected check's requirements with the declared set before execution and turns an unmet requirement into `skipped`/`unsupported_capability` — the selection is honoured, the gap is visible, and nothing is reported as a pass it did not earn.

An unknown capability **member** is a different matter and stays exit 2: a name outside the thirteen is wrong whoever selects what, so it is a configuration fault like any other.

## 9. Options that override the file

An option replaces one resolved value for one invocation; the configuration file is never rewritten.

| Option | Overrides |
|--------|-----------|
| `--config PATH` | which file is read at all (default `qrcheck.toml`) |
| `run --adapter NAME` | `default_adapter` |
| `run --seed N` | `seed` |
| `run --report-dir PATH` | `report_dir` |
| `run --ec <level>` | nothing — the error-correction level is per-run and has no configuration key; it defaults to `M` |
| `loc --limit N` | `loc_limit` |
| `loc --path PATH` | nothing — the tree to count defaults to the directory holding the configuration file |
| `corpus build --seed N` | `seed` |
| `corpus build --out PATH` | `corpus_path` |
| `corpus build --force` | nothing — it permits replacing a differing destination that regeneration would otherwise leave untouched |

The selection options — `run --checks`, `run --class`, `run --fail-fast` — have no configuration counterpart at all; they choose what runs, not how the product is configured. `README.md` owns the full command surface, and `--checks` and `--class` semantics belong to it.


## 10. The committed default, in full

The committed `qrcheck.toml`, with the comments that explain each key in the file itself omitted here so the values stand out. Every value in this block is the value in that file; a change to one belongs in both, in the same commit, and the two must stay comparable value for value — parsing each with `tomllib` and comparing the resulting mappings is the whole check.

```toml
default_adapter = "reference"
report_dir = "reports"
seed = 20240101
corpus_path = "corpus/default.jsonl"
loc_limit = 60000

[targets.reference]
type = "reference"
timeout_s = 30.0
contrast_floor = 7.0

[targets.defective]
type = "defective"
timeout_s = 30.0
contrast_floor = 7.0

[render]
svg_rasterizer = "auto"

[checks.performance]
encode_p95_ms = 250.0
min_symbols_per_minute = 120.0
encode_samples = 100
decode_p95_ms = 400.0
decode_samples = 200
decode_image_px = 500
warmup = 5

[checks.damage]
occlusion_margin = 0.05
blur_radius_per_module = 0.6
jpeg_quality = 40
noise_sigma = 12.0
rotation_deg = 15.0
perspective_deg = 10.0
downscale_px_per_module = 3.0
logo_area_fraction = 0.15
```

Neither built-in target declares a `capabilities` list, and that is deliberate rather than an omission: the `reference` and `defective` adapters declare all thirteen capabilities themselves, so the empty default would add nothing — and a reference run can then be required to finish with zero skips.

## 11. Three edits and their exact outcomes

**Adding a `cli` target** for a script that takes a payload and an output path:

```toml
[targets.mygen]
type = "cli"
command = ["python3", "examples/cli_target.py"]
payload_arg = "{}"
output_arg = "--out={}"
returns = "file_path"
```

`qrcheck run --adapter mygen` then exercises it; `default_adapter` still selects `reference` unless it is changed too. Declare `capabilities` as well: an omitted list is an empty set, so every check needing `png` reports `skipped`. `targets.md` walks through the same declaration field by field, including the `library` and `http` forms.

**A `returns` typo** — `returns = "bytes"` instead of `bytes_path` — is exit 2, no report, and the diagnostic names the key and the four members it accepts:

```
qrcheck run: configuration error: targets.mygen.returns: invalid value 'bytes'; allowed: ['base64', 'bytes_path', 'file_path', 'raw_bytes']
```

`run`, `doctor`, `corpus build` and `loc` all read the document and all fail the same way. `qrcheck list` reads no configuration and is unaffected.

**A `capabilities` list that omits `svg`** while `RN-SVG` is selected does not fail the run: that check's cases are `skipped` with reason `unsupported_capability`, listing `svg` as the requirement to add. `counts.skipped` increments, `counts.passed` does not, and the exit code is unmoved — a run of passes and skips exits `0`. The gate that demands zero skips in a reference run is what catches it, which is exactly why that gate exists.

