# Declaring a target

How to put a QR generator behind `qrcheck` without touching the framework's assertions. A target is one `[targets.<name>]` table in the configuration file, chosen at run time by `qrcheck run --adapter <name>` or by the top-level `default_adapter`. Its `type` selects the adapter that drives it: `library`, `cli` and `http` are configurable, `reference` and `defective` are built in.

This document is written for the person wiring a generator in. For every key it says what you must respect, what the adapter does with the value at run time, and what each failure you can cause means and how to fix it. The tables that name a file under `examples/` are copied from the shipped scripts, because `tests/integration/test_cli_adapter.py` and `tests/integration/test_http_adapter.py` execute exactly those tables — they are the configuration CI keeps working, not an illustration of one.

Where the rest of the contract lives: `configuration.md` owns every key's type, default, validation rule and exit-2 diagnostic, so this document states the rule you must respect and sends you there for the exhaustive table. `architecture.md` owns the layering rationale, the adapter port and the error taxonomy in full. `checks.md` says what each check asserts, `dependencies.md` pins the versions, and `performance.md` the budgets. Nothing here restates a number from any of them.

## 1. The shape of a target declaration

A declaration answers two questions — **where** the generator is and **how to talk to it** — and never a third. It carries no threshold, no seed, no check selection and no assertion: what is asserted belongs to `qrcheck/checks/`, and a target table that could change an assertion would be a table that can make a check pass by relaxation. Every check reaches your generator through one port (`qrcheck/targets/base.py`), so nothing about your generator's shape reaches the checks.

Four keys every target shares, whatever its `type`:

| Key | Type | Required | Meaning | Rule |
|-----|------|----------|---------|------|
| `type` | string | yes | which adapter drives this target | one of `library`, `cli`, `http`, `reference`, `defective` |
| `timeout_s` | number | no | ceiling on one operation against the target, not on the run | `> 0`; exceeding it raises `TargetTimeout` |
| `contrast_floor` | number | no | the luminance ratio this target undertakes to render | `> 1.0`; `RN-CONTRAST` asserts at or above it and `IV-BAD-OPTION` builds a request measured below it |
| `capabilities` | array of strings | no | the operations the target can perform | every member one of the thirteen names in section 8 |

The three configurable kinds differ in one respect that matters when you write the table: which per-type keys are legal. A key belonging to another kind — `command` on a `library` target, `callable` on an `http` target — is an **unknown key**, not an inert one, and configuration loading exits 2 naming its dotted path. The per-key rules and their exact diagnostics are in `configuration.md` sections 4 and 5.

| `type` | Where the generator is | What the adapter owns |
|--------|------------------------|-----------------------|
| `library` | a Python callable in this interpreter | the import, the keyword arguments, the return-value decoding |
| `cli` | a program the framework runs once per request | the argv assembly, the subprocess boundary, the exit-status interpretation |
| `http` | a service on the loopback interface | the request body, the response parsing, the image extraction |

`reference` and `defective` take only the four shared keys. They are the framework's own self-check pair — a conforming segno-backed generator and one with five injected defects — and configuring an input on them would be a key with no effect, so it is an error.

## 2. `type = "library"` — an in-process import and call

The strongest form of the no-vendor-in-the-framework rule: the adapter names no generator library and imports none, so pointing `qrcheck` at a different generator is a table rather than a code change.

```toml
[targets.acme_lib]
type = "library"
callable = "acme_qr.render:png_symbol"
returns = "raw_bytes"
timeout_s = 30.0
capabilities = ["png", "ec_level", "module_size", "color"]

[targets.acme_lib.kwargs_map]
payload = "data"
ec_level = "level"
module_size_px = "scale"
foreground_rgb = "dark"
background_rgb = "light"

[targets.acme_lib.metadata_map]
version = "symbol.version"
ec = "symbol.level"
```

`acme_qr.render` is the module you already have; it is not part of this repository. The table above calls `png_symbol(data=…, level=…, scale=…, dark=…, light=…)` once per request, and because `metadata_map` is configured the callable answers with a two-element tuple — the image value and a mapping — rather than the image value alone:

```python
# acme_qr/render.py — the target's own code, unchanged by qrcheck
def png_symbol(data, level, scale, dark, light):
    png_bytes = ...                    # whatever the generator does
    return png_bytes, {"symbol": {"version": 3, "level": level}}
```

| Key | Type | Required | Meaning | Rule |
|-----|------|----------|---------|------|
| `callable` | string | yes | `module:attribute` of the generator | exactly one `:`, both parts non-empty, every dotted part a Python identifier; a dotted attribute (`module:Factory.build`) is walked segment by segment |
| `kwargs_map` | table of strings | yes | framework request field to the callable's keyword argument | keys are request field names — `payload`, `ec_level`, `output_format`, `module_size_px`, `quiet_zone_modules`, `foreground_rgb`, `background_rgb`, `logo_overlay_path`, `version`, `mask` — never the template tokens of section 5; two fields on one kwarg is refused |
| `returns` | string | yes | how the returned value carries the artefact | one of `bytes_path`, `file_path`, `base64`, `raw_bytes` (section 6) |
| `metadata_map` | table of strings | no | which metadata the callable reports, and where | keys `version`, `ec`, `mask`, `mode`; declaring it changes the return shape to a two-element tuple (section 7) |

The `callable` path is checked for **syntax** when the configuration loads, so the same document stays valid on a machine where the module is not installed; it is resolved **lazily**, on the first `generate`, which is why `qrcheck doctor`'s single probe — not the adapter's construction — is what proves it exists. A missing module or a missing attribute raises `TargetUnavailable`; any other exception the module's own import body raises belongs to the target module and propagates as a framework error rather than being misreported as an unreachable target.

**Only mapped fields are passed.** A field you leave out of `kwargs_map` is never inferred, so a generator whose error level is fixed at M simply maps no `ec_level` and is never handed a value it did not ask for. `payload` arrives as the request's exact bytes and is never re-encoded on your behalf; the other fields arrive with the request's own types — `module_size_px` as an integer, `foreground_rgb` as a three-integer tuple.

At run time the adapter, in this order: refuses the request if it asks for a capability this target did not declare (before the import, so nothing is touched); imports and resolves the callable; calls it once with the mapped keyword arguments under a daemon-thread watchdog bounded by `timeout_s`; decodes the returned value through `returns`; and projects the metadata mapping through `metadata_map`. A call that outlives `timeout_s` cannot be interrupted inside one interpreter, so it is abandoned — it keeps running with its result discarded, it is never joined, and the case reports `TargetTimeout`.

## 3. `type = "cli"` — an out-of-process command

The generator is a program the framework runs once per request. `command` is untrusted configuration and the payload is untrusted corpus data, so the command line is built as a list of strings and run without a shell: a payload carrying `;`, `$(…)` or a quote is one literal argv word, not a command.

This is the table `examples/cli_target.py` carries in its own docstring and the one `tests/integration/test_cli_adapter.py` loads through the real configuration loader:

```toml
[targets.example_cli]
type = "cli"
command = [".venv/bin/python", "examples/cli_target.py"]
payload_arg = "--payload-b64={payload_b64}"
output_arg = "--output={}"
returns = "file_path"
timeout_s = 30.0
capabilities = ["png"]
```

Every value there is the script's own requirement rather than a sketch: `--payload-b64` is the flag `examples/cli_target.py` reads and `{payload_b64}` — not a bare `{}` — is the form its strict base64 decode accepts, so a bare `{}` beside that flag would hand the script raw UTF-8 text, which it refuses as invalid base64. `.venv/bin/python` is named rather than a bare `python`, because the child needs the interpreter the dependencies are installed in: a child that cannot import its own generator exits 1 with a traceback, and the adapter reports that as a contract error, which is the correct classification — an interpreter without the dependency set is not a symbol the framework may judge. The script's own docstring carries the same block, and that integration test executes it, so a change to either the script's arguments or this table breaks CI.

| Key | Type | Required | Meaning | Rule |
|-----|------|----------|---------|------|
| `command` | array of strings | yes | the literal argv prefix of every invocation | non-empty, every word a non-empty string that carries no placeholder: words are passed literally and **never rendered**, so a placeholder in one is exit 2 naming `targets.<name>.command` and its index, a placeholder belongs in `payload_arg`/`output_arg`, and a fixed value belongs here as a literal word; the child inherits the framework's environment and working directory |
| `payload_arg` | string | yes | how the payload reaches the target | contains `{}`, or starts with `-` (section 5) |
| `output_arg` | string | yes | where the artefact is expected | contains `{}`, or starts with `-`; a `{}` word carries the path of the file the framework names for the target to write |
| `returns` | string | yes | how the artefact is read back | one of `bytes_path`, `file_path`, `base64`, `raw_bytes` (section 6) |
| `metadata_map` | table of strings | no | which metadata the target reports, and where | keys `version`, `ec`, `mask`, `mode`; refused beside a stdout-based `returns` (section 7) |

For a script whose own flags are `--payload` and `--out`, the two templates are `payload_arg = "--payload={}"` and `output_arg = "--out={}"`, with `returns = "file_path"` because the target writes the file the framework named. Use `{payload_b64}` instead of the bare `{}` when that flag decodes base64: a bare `{}` beside a base64 flag hands the script raw text, which a strict decoder refuses.

**What the adapter does at run time.** It first refuses a request that asks for a capability this target did not declare, or for a version pin, a mask pin or a logo overlay that no template in this transport can carry (section 8), before any process exists. Then it creates a temporary directory under `tempfile.gettempdir()` whose name carries the product-wide prefix `qrcheck-`, names the artefact `symbol.png` or `symbol.svg` inside it according to the format the request asks for — the extension is how a target that infers its writer from the path learns the format when your configuration has no `{format}` placeholder — and builds the argv as the words of `command`, then the payload word, then the output word. It runs that argv with `shell=False` and `timeout_s` as the subprocess timeout, capturing stdout and stderr, and removes the temporary directory in a `finally` block, so it survives neither a timeout nor a non-zero exit nor a contract error. A leaked `qrcheck-` directory would fail `CL-ARTIFACTS`, a check that has nothing to do with this file, which is why the prefix is a contract rather than a private habit.

**How an exit status is read.** Exit 0 means the artefact is where `returns` says it is. A non-zero exit is a **declared rejection**, and therefore `TargetRejected` carrying your text verbatim, only when the last non-empty line of stderr or of stdout is a JSON object with a string member `error` — `{"error": "payload too long"}` — or a line beginning with `rejected:`. Every other non-zero exit is `TargetContractError` carrying the child's captured output, because a rejection you did not declare must not count as one: the `IV-*` checks pass on a declared rejection, and a fabricated one would make them vacuous.

**What a `cli` target cannot convey.** A template produces exactly one argv word, so a configuration conveys the payload and the artefact path and nothing else. Put fixed values in `command` as literal words — `["--ec", "Q", "--size", "6"]` — if your generator needs them; those then apply to every request, and the run's own error level never reaches the target. Either way a statically fixed constant is not a capability this configuration conveys, so the `capabilities` list above spends both words on the payload and the artefact path and declares `png` alone. Declaring `module_size` or `color` there would claim an operation the table cannot transmit, and a check that needed it would measure the script's defaults and blame the target for the difference.

## 4. `type = "http"` — a loopback service

The generator is a service the framework calls over the loopback interface. The adapter knows no vendor's endpoint, field name or response key: the URL, the method, the request body, the path to the image inside the response, the way that image is carried and the paths to any metadata all come from this one table, so a fourth vendor is a configuration edit rather than a code change. That is the framework-side rule stated from your side: **the vendor's response shape is yours to describe here, and no part of it exists anywhere in the framework.**

These are the two tables `examples/http_target_server.py` carries in its own docstring and `tests/integration/test_http_adapter.py` drives through the real configuration loader:

```toml
# base64 envelope: image_field is a dotted path into the JSON response
[targets.example_http]
type = "http"
url = "http://127.0.0.1:PORT/generate"
method = "POST"
image_field = "image_b64"
image_encoding = "base64"
timeout_s = 30.0
# No version_pin or mask_pin: the eight template placeholders carry no version or
# mask, so declaring either would make a pin-needing check resolve as runnable and
# only skip mid-run instead of skipping before execution.
capabilities = [
    "png", "ec_level", "module_size", "quiet_zone", "color",
    "metadata_version", "metadata_ec", "metadata_mask", "metadata_mode",
]

[targets.example_http.request_template]
payload_b64 = "{payload_b64}"
ec_level = "{ec_level}"
format = "{format}"
module_size = "{size}"
quiet_zone = "{margin}"
foreground = "{fg}"
background = "{bg}"

[targets.example_http.metadata_map]
version = "metadata.version"
ec = "metadata.ec_level"
mask = "metadata.mask"
mode = "metadata.mode"
```

```toml
# raw body: image_field = "_body", image_encoding = "raw"
[targets.example_http_raw]
type = "http"
url = "http://127.0.0.1:PORT/raw"
method = "POST"
image_field = "_body"
image_encoding = "raw"
# No metadata_* capabilities: the raw route returns the PNG with no JSON envelope,
# so there is no metadata to map (the block declares no map).
capabilities = ["png", "ec_level", "module_size", "quiet_zone", "color"]

[targets.example_http_raw.request_template]
payload_b64 = "{payload_b64}"
ec_level = "{ec_level}"
format = "{format}"
module_size = "{size}"
quiet_zone = "{margin}"
foreground = "{fg}"
background = "{bg}"
```

**`PORT` is the port the server printed, and nothing else.** `examples/http_target_server.py` binds the literal address `127.0.0.1` on an ephemeral port, writes that integer as the one and only line it ever puts on stdout, and logs to stderr; there is no option, argument or environment variable for host or port, so the process cannot be pointed at a non-loopback interface. Substitute the integer it printed when you drive it by hand; a test reads it from the process instead of hard-coding one. The `[targets.<name>.request_template]` and `[targets.<name>.metadata_map]` sub-tables are sent as exactly the JSON object and the mapping their keys spell, and the arrays and tables above are wrapped to the repository's 100-column limit, which changes no TOML or JSON semantics.

| Key | Type | Required | Meaning | Rule |
|-----|------|----------|---------|------|
| `url` | string | yes | the endpoint, per request | scheme `http` or `https`, host one of `127.0.0.1`, `localhost`, `::1`, and no `user:password@` prefix |
| `method` | string | yes | how the substituted template is carried | `GET` or `POST` |
| `request_template` | table, or a string holding that object's JSON | yes | the request body, with placeholders | every string leaf's field name drawn from the eight in section 5, and at least one payload placeholder |
| `image_field` | string | yes | where the image sits in the response | `_body` when the response body *is* the image, otherwise a dotted path into the parsed JSON (`data.0.image`) |
| `image_encoding` | string | yes | how that value carries the image | `base64` (base64 text) or `raw` (a string's UTF-8 bytes, or a byte value, unchanged) |
| `metadata_map` | table of strings | no | which metadata the response carries, and where | keys `version`, `ec`, `mask`, `mode`; a target whose `image_field` is `_body` has no envelope, so it declares no map |

**The host rule is the offline constraint's enforcement point.** `qrcheck` opens exactly one kind of socket — to a target it was told about — and refuses to be aimed anywhere but the loopback interface: a `url` naming any other host exits 2 naming that host, and a hosted generator therefore cannot be declared at all. The allow-list is one list read by both the configuration schema and the adapter, so the boundary cannot widen on one side while staying closed on the other; the adapter re-checks it at construction as well, creates its client with `follow_redirects=False` and `trust_env=False` so neither a redirect nor an `HTTP_PROXY` variable can carry a request off loopback, and reads no credential, cookie jar or authorisation header.

**What the adapter does at run time.** It refuses a request that asks for a capability this target did not declare, or for a version pin, a mask pin or a logo overlay that no placeholder can carry, before any socket is opened. Then it substitutes the template — a string leaf that is one whole placeholder receives the value with its own type, so `{size}` arrives as a JSON number; a placeholder embedded in a longer string renders as text; a placeholder whose request value is `None` omits its containing key rather than sending an empty string; `{{` and `}}` are literal braces — and sends the result. `GET` carries it as query parameters, so every leaf must be a scalar. Any other method sends it as a JSON body. The client is created on the first `generate` and reused for the run.

A non-2xx status is always `TargetContractError`, never `TargetRejected`: this transport has no declared-rejection channel, so a refusal and a crash are reported alike. That is why the `IV-*` checks are never pointed at an HTTP target — they pass only on a declared rejection — and it is a property of the transport rather than an oversight. Response bodies are read once under a 16 MiB bound, so a runaway target costs one contract error instead of this process' memory.

## 5. The request-template placeholders

`request_template` may interpolate these eight names and nothing else:

| Placeholder | Substituted with |
|-------------|------------------|
| `{payload_b64}` | the payload bytes, base64-encoded |
| `{payload_utf8}` | the payload as UTF-8 text |
| `{ec_level}` | the requested error-correction level, `L` / `M` / `Q` / `H` |
| `{format}` | the requested output format, `png` or `svg` |
| `{size}` | the requested module edge in pixels, when the request pins one |
| `{margin}` | the requested quiet zone in modules, when the request pins one |
| `{fg}` | the requested foreground colour, as `#rrggbb` |
| `{bg}` | the requested background colour, as `#rrggbb` |

That is the complete set: a ninth placeholder does not exist, and one appearing in a template exits 2 naming the offending leaf with the allowed list beside it. A placeholder may carry a conversion or a format specification — `{size:d}`, `{payload_utf8!s}` — and only the field name is checked.

Both payload channels are real and the shipped HTTP example answers either one. A template sending `payload_b64 = "{payload_b64}"` carries any bytes; one sending `payload_utf8 = "{payload_utf8}"` carries readable text and refuses a payload that is not valid UTF-8. Choose by what your corpus must survive — a payload with an embedded NUL or a byte that is not UTF-8 travels only in the base64 channel.

Two consequences follow from the set being closed. `{payload_utf8}` disappears from a request whose payload is not valid UTF-8, and a template whose only payload channel is that placeholder refuses such a payload as `TargetUnsupported` rather than sending a request with no payload at all. And there is deliberately no placeholder for a version pin, a mask pin or a logo overlay, which is why a request that pins one is refused before any I/O — including on a `cli` target whose argument templates use the same vocabulary.

The `cli` adapter's `payload_arg` and `output_arg` accept those eight names **plus** the bare `{}`, which means "the payload here" in `payload_arg` and "the artefact path here" in `output_arg`. The bare form carries the payload as UTF-8 text and raises `TargetContractError` for a payload that is not strictly UTF-8-decodable, so a binary payload needs `{payload_b64}`. A template that carries no placeholder and starts with `-` is passed as **one** argv word and the raw payload bytes are written to the child's stdin — the only way to deliver a payload no argv word can hold — which is why the stdin spelling is `--payload-b64=-` (the target reads stdin for the value `-`) or `--payload-stdin` (a flag taking no value), never the two-word `--payload-b64 -`: the adapter never splits a template into two words, so that configuration would pass the flag with no value and the target would reject its own usage. An unknown name in either template is exit 2 naming `targets.<name>.payload_arg` or `targets.<name>.output_arg`, with the allowed list beside it, so a typo is refused while the document is read rather than once a check is already running.

## 6. `returns`, and how a response becomes a symbol

`returns` is the one key that says how the artefact comes back. All four values converge on a single decoding function, so the `library` and `cli` adapters accept and refuse exactly the same bodies.

| `returns` | What the adapter expects to find | Not found |
|-----------|----------------------------------|-----------|
| `raw_bytes` | the artefact itself, as `bytes` (a `bytearray` or `memoryview` is normalised) | a value that is not bytes-like is `TargetContractError` |
| `base64` | base64 text, as a `str` or ASCII `bytes`, decoded strictly so a stray character is a failure rather than silently skipped | undecodable text is `TargetContractError`; base64 that decodes to nothing is an empty artefact, not a zero-length image |
| `bytes_path` | a path the **target** chose, holding the artefact; on a `cli` target, the path the target reports on the last non-empty line of its stdout | an unreadable path, a directory, or a text path that is empty is `TargetContractError` |
| `file_path` | a path the **framework** supplied — the file it named through `output_arg` — which the target then wrote | as for `bytes_path`; on a `library` target both path kinds read the path the callable returns, because that transport has no framework-supplied output path in its vocabulary |

Whichever route the bytes take, one gate then confirms they plausibly are the artefact they claim to be: an empty body, a body that is neither PNG, JPEG nor SVG, or a body recognisably a **different** format from the one the request asked for is a `TargetContractError`, and the message carries the leading bytes' hex. At the default limit of 256 bytes that is 512 hex characters, and the same prefix travels on the error so the evidence field `hex_prefix` records what actually arrived rather than the diagnostic built from it. Whether the bytes are a *valid* image is a later decision, made by the decode layer, and is not duplicated here.

What a target returns becomes a `GeneratedSymbol`, and three of its fields have a status you must know:

- `image_bytes` is the artefact **exactly as the target produced it** — never re-encoded, resized or repaired by the framework, so the image in the evidence directory is the one a failure is about.
- `output_format` is the format actually produced, and it must match the request. A target that answers a `png` request with SVG bytes has broken its contract, and that is where it is caught.
- `declared_version`, `declared_ec`, `declared_mask` and `declared_mode` are the target's own claim and are **evidence only, never the assertion**. `SY-VERSION-MIN` takes the version from the sampled module count, `SY-EC-MATCH` reads the error level from the primary decoder and from the symbol's format information, and `SY-MASK` reads the mask from the format information: a target that lies about its metadata is recorded as having lied, not believed.

## 7. `metadata_map` and dotted paths

`metadata_map` maps a **framework** field name to the dotted path carrying that field in your response, so no vendor's key is ever written into an adapter:

```toml
# the envelope route's metadata paths, as configured in section 4
[targets.example_http.metadata_map]
version = "metadata.version"
ec = "metadata.ec_level"
mask = "metadata.mask"
mode = "metadata.mode"
```

`version = "meta.version"` reads `{"meta": {"version": 3}}`. A path segment is read as a mapping key when the current value is a mapping and as a zero-based index when it is a sequence, so `"images.0.png"` reads key `images`, then element `0`, then key `png`. The four framework fields are exactly `version`, `ec`, `mask` and `mode`, and any other key is a configuration error rather than a field the framework could store: it exits 2 naming the unknown field and listing the four.

One implementation resolves these paths for both transports — `resolve_dotted_path` and `apply_metadata_map` in `qrcheck/targets/base.py` — so a configured source means the same thing for a command target and an HTTP one. They differ only in where the object comes from: an `http` target's parsed JSON response, or the single JSON object a `cli` target writes to stdout. That is why a `metadata_map` configured beside a stdout-based `returns` (`base64` or `raw_bytes`) is refused when the adapter is built: the artefact and the metadata would both have to come from the same stream, and the adapter will not guess which.

A path that resolves to nothing — a missing segment, a `null` value, or a value that cannot be coerced to the field's type — is the "configured field the response does not carry" case: `TargetContractError` naming the framework field, the configured path, and the keys the response did offer. Metadata is never used to pass a check, so a mapping that is wrong costs you a contract error rather than a silent pass.

## 8. Declaring `capabilities`

`capabilities` is what your target says it can do. The set is closed at thirteen members, and a declared capability must be **true**: the value is read by the runner before any check runs, so an optimistic list produces failed checks you cannot explain.

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
| `metadata_version` | report the symbol's version in its metadata |
| `metadata_ec` | report the symbol's error-correction level in its metadata |
| `metadata_mask` | report the symbol's mask in its metadata |
| `metadata_mode` | report the symbol's encoding mode in its metadata |

Which checks each member unlocks is the other half of the same contract: `RT-ROUNDTRIP` requires `png`; `SY-EC-MATCH` requires `png` and `ec_level`; `RN-QUIET-ZONE` requires `png` and `quiet_zone`; `SY-MATRIX-EQUIV` requires `png`, `ec_level`, `version_pin` and `mask_pin`; `RN-SVG` requires `svg` and `quiet_zone`. `qrcheck list --json` prints the exact requirement set of every check on this installation, which is the authoritative answer for any of them.

**Under-declaring is a skip, not an error.** A selected check whose requirements your declared set does not cover reports `skipped` with reason `unsupported_capability`, counted separately in `summary.json` and never counted as a pass. A shortfall is not a configuration error because whether a set is sufficient depends on the run-time `--checks` selection: the same file would be wrong for one selection and correct for the next. Only an unknown **member** exits 2, because a name outside the thirteen is wrong whoever selects what.

**Over-declaring is not forgiven.** The capability gate reads the declaration, so a request for an operation you declared but do not perform reaches your generator, and what comes back is your target's own rejection or contract failure — reported against your target, in evidence you cannot argue with. The gate an adapter applies before a call is also narrower than the declared set: every request is gated on its output format, on `module_size`, `quiet_zone` or `color` only when the request pins one of those, and — on a `library` target, whose keyword arguments can carry them — on `version_pin`, `mask_pin` or `logo_overlay` when the request asks. `ec_level` and the four `metadata_*` members are never gated, because every request carries a level and metadata is reported rather than requested; what reads the full declaration is the registry, which intersects each selected check's requirements before execution. That is why declaring `ec_level` on a target that genuinely honours the level is what lets `SY-EC-MATCH` run instead of skip.

**Three capabilities cannot be conveyed by a template, wherever you declare them.** No placeholder in a `cli` argument template or an `http` request template carries a version pin, a mask pin or a logo overlay, so a request that asks for one is refused with `TargetUnsupported` before any I/O even if you declared `version_pin`, `mask_pin` or `logo_overlay`. Declaring them anyway only moves the skip from before execution to the middle of it — which is why the shipped HTTP example declares neither pin.

The two built-in self-check targets declare all thirteen themselves, which is what lets the clean-run gate demand zero skips: a `reference` run with a skip would be a gate with a hole in it.

## 9. What each of the five adapter errors means for you

An adapter raises nothing but these five, and the framework's whole reading of your target's failure goes through them. The taxonomy and its single-trigger rule are in `architecture.md`; what follows is what each one says about your table.

| Error | What the target did | What to do instead |
|-------|--------------------|--------------------|
| `TargetUnavailable` | the process could not be spawned, the connection was refused, or the configured import path is missing | make the target startable and the path correct: the interpreter named in `command`, the module the `callable` names, the service the `url` dials |
| `TargetTimeout` | no response arrived within `timeout_s` | make the generator answer, or raise `timeout_s` — a configuration decision, not a code change; a `library` call that overruns keeps running with its result discarded, so treat the timeout as a real ceiling |
| `TargetRejected` | the target declared a client-side rejection of a request the framework considers valid | keep declaring rejections, and in the documented encoding; the text is reported verbatim because `IV-EMPTY` and `IV-BAD-OPTION` pass only on a rejection the target itself declared |
| `TargetContractError` | a response arrived but was unusable: an empty body, undecodable base64, image bytes no image library can open, a format other than the one requested, a field named in configuration that the response does not carry — and, on `http`, any non-2xx status | fix the response shape mapping: the `returns` kind, the `image_field` path, the `metadata_map` paths, the format your generator actually writes |
| `TargetUnsupported` | the request needed a capability the adapter did not declare, or a version pin, mask pin or logo overlay that no template can convey; raised before any I/O | declare the capability you do honour, and leave undeclared the ones this transport cannot carry |

How each one lands in the report: `TargetUnavailable` and `TargetTimeout` become an `error` result rather than a failed one, because the framework never obtained a symbol to judge; `TargetContractError` fails the case; `TargetUnsupported` skips it; and `TargetRejected` is the check's decision — a pass where a declared rejection is the expected outcome (`IV-EMPTY` and `IV-BAD-OPTION`) and a failure everywhere else.

How a rejection reaches the framework depends on the transport, and each is the only channel that exists for it: a `library` callable declares one by raising an exception whose text begins `rejected:`; a `cli` target declares one with a JSON object carrying a string member `error`, or a line beginning `rejected:`, on the last non-empty line of stderr or stdout beside a non-zero exit; an `http` target has no rejection channel at all.

The rule that follows for you: **a target that raises an arbitrary exception has broken the contract.** The framework does not translate it into a target failure — it records an internal error, with the traceback, and the process exits 3 at the end of the run. There is no sixth error type, and raising something outside the five does not reach the report as a rejection, a timeout or a contract failure.

## 10. Verifying a declaration before a full run

Three commands, in this order, tell you whether the table is right before a 1,283-case run does.

**`qrcheck doctor --adapter <name>`** verifies the configuration, the interpreter, the target's reachability and the local toolchain, reporting every prerequisite rather than stopping at the first fault. It asks your target for exactly one ordinary probe symbol — nothing reads it; its arrival is the whole assertion. Exit 0 means every blocking prerequisite passed, exit 2 that one failed, exit 3 an unexpected framework fault. `--json` prints one object with exactly six keys:

| Key | Content |
|-----|---------|
| `adapter`, `type` | the configured name and the kind the table declares |
| `reachable` | whether the probe generate produced a symbol |
| `capabilities` | the declared members, sorted |
| `dependencies` | one entry per prerequisite, each with `name`, `version`, `status` and `blocking`; `status` is `ok`, `missing` or `degraded` |
| `svg_rasterizer` | the rasterizer `RN-SVG` would use: an object with `configured`, `name`, `available` and `attempted` — the discovery order actually probed — or `null` when the configuration named none |

A primary decoder, an oracle or a core library that is missing or cannot be imported is blocking — exit 2, the report names the dependency and the pinned lock file. A missing secondary decoder or SVG rasterizer is **degrading**: exit 0, the run proceeds in single-decoder mode or skips `RN-SVG` with reason `missing_dependency`, and the report states the degradation rather than leaving a reader to infer it from a skip.

**`qrcheck list --json`** prints every registered check with exactly five keys — `id`, `title`, `family`, `requires` and `dependencies` — sorted by `id`. Cross the `requires` of each check against your `capabilities` list to see which checks your target can attempt and which will skip.

**`qrcheck run --adapter <name> --checks RT-ROUNDTRIP`** is the narrow end-to-end proof: one check against the real target, real decoders and real reports. Add `--report-dir <path>` to keep that run's output apart from the next one. A failing case leaves its image, the request that produced it and the decoded-or-error outcome under `<report-dir>/evidence/<check-id>/<failure-id>/`, which is usually all you need to see what your generator actually emitted.

The exit codes a target author meets:

| Exit | Meaning |
|------|---------|
| `0` | the run completed with no failure and no error; a skip does not make it a failure, but a skip is also not a pass |
| `1` | at least one check failed and none errored |
| `2` | configuration, selection, corpus filter, an unusable pinned dependency (primary decoder, oracle or core library) or target availability — decided before any check runs, and **no report is written**; the offending key, filter or dependency is named |
| `3` | a check errored internally, the report directory could not be written, or an unhandled framework exception occurred |

## 11. Adding a fourth adapter kind

If your generator needs a transport none of the three provides, the seam is designed for it. The edits are these and no others:

1. **One module** — `qrcheck/targets/<kind>.py` — implementing the `Adapter` port: `name`, `capabilities`, `generate(request) -> GeneratedSymbol` and `close()`. Declare the capabilities from the resolved record, call `require_capabilities(...)` at the top of `generate` before any I/O, and convert every exception your library raises into one of the five adapter errors, leaving nothing else to escape.
2. **One entry** in `ADAPTER_TYPES` in `qrcheck/targets/__init__.py`, mapping the new `type` value to your module and class. That table is the single way an adapter is ever built, and its rows import lazily, so adding one costs nothing to processes that never build an adapter.
3. **One configuration value** — the new `type` in `AdapterType` in `qrcheck/config/models.py` — plus that kind's per-type keys in the table in `qrcheck/config/schema.py`, because a `type` with no keys of its own is a kind nothing can configure.
4. **Stop.** No file under `qrcheck/checks/` changes.

That last line is the property the extension point exists to preserve, and it is held mechanically rather than by convention. A check sees only the port, so `tests/unit/test_layering.py` walks the AST of every module in the check layer and fails on any import of a concrete adapter — `library`, `cli`, `http`, `reference` or `defective` — and it fails just as loudly on any module under `qrcheck/targets/` importing `qrcheck.checks`. `tests/unit/test_adapter_types.py` closes the other end: it asserts that the factory's keys are exactly the declared `type` values, that each entry resolves to the class the extension model names, that each entry given a record of its own kind returns an object satisfying the port (by `isinstance`, never `issubclass`, because the protocol has non-method members), and that an undeclared type raises `KeyError` instead of resolving to `None`.

A new **check** is the symmetric change: one family module registration plus the capability requirements it declares, and no adapter change at all. Neither side of the seam ever sees the other, which is what makes both directions cheap.

## 12. What not to do

- **Do not import `qrcheck` from a target.** Both shipped examples import nothing from the framework, deliberately: a target that leans on the code testing it hides exactly the coupling this contract forbids, and it will not survive the framework's own refactoring.
- **Do not put a vendor's field names, endpoint or response shape anywhere but configuration.** If a key of your API appears in a Python file, the table has stopped being the description of your target and a code change has become necessary to change vendors — which is the one thing the adapters exist to prevent.
- **Do not declare a capability you do not honour.** An honest short set costs you skips you can explain; an optimistic one costs you failures you cannot.
- **Do not invent a failure.** There are five adapter errors, each with one trigger, and no sixth; a rejection you did not declare, a timeout you did not take, or an exception you let escape will be read as a contract error or a framework defect, never as the outcome you intended.
- **Do not add a retry, a sleep or a warm-up loop to hide flakiness.** Each case invokes the adapter a declared number of times and the suite asserts those counts, so a retry surfaces as the count plus one and fails the run. An intermittent QR check is a defect to fix, not to suppress.
