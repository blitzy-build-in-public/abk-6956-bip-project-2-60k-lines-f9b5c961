# qrcheck dependencies and platforms

Every package `qrcheck` installs, the two hash-locked requirement files that own their versions, and what each absence costs on which platform. Read this when you are adding, removing or regenerating a pin, or when you are diagnosing a host on which the tool will not start.

Two rules govern the whole document. Every dependency is pinned to an exact version in a committed lock file and installed with `--require-hashes`, so no range, no `latest` and no `~=` exists anywhere in the product. And the dependency set is **closed**: a new runtime dependency is a defect unless the tables below already list it (§10 says why).

What each dependency is *used for* downstream is owned elsewhere: `architecture.md` §9 for what the decode pipeline does with a missing decoder, `configuration.md` §6.1 for the `[render]` key that selects a rasterizer, `performance.md` for the environment the timing floors are defined against, `checks.md` for the assertions each tool feeds, `targets.md` for declaring a target.

## 1. The two locks, and which file is installed

| File | Role | Who installs it | Sole authority for |
|------|------|-----------------|--------------------|
| `requirements.in` | Compile input: the seven direct runtime pins | Nobody directly — input to `pip-compile` | The direct runtime pins |
| `requirements.lock` | The runtime closure, hash-locked | CI, and every user | Every runtime version and hash, transitive ones included |
| `requirements-dev.in` | Compile input: `-r requirements.in` plus the five development pins | Nobody directly — input to `pip-compile` | The development pins |
| `requirements-dev.lock` | The runtime closure plus the development closure and the unsafe block, hash-locked | The `unit` and `e2e` CI jobs, and a contributor running the gates | Every development and transitive version and hash |

**A user installs the runtime lock; a contributor adds the dev lock.** The dev lock is a superset that installs on its own, because it starts from `-r requirements.in` — so it cannot disagree with `requirements.lock` about a shared transitive pin.

`pyproject.toml` declares the same seven runtime pins in `[project].dependencies`, sorted by canonical name, so an editable install cannot drift from the lock. It carries no `[project].optional-dependencies`: a `dev` extra would restate the development set the dev lock already owns, and the five development pins must never appear in `[project].dependencies` or in an extra.

Never hand-edit a lock, with two documented exceptions: the conditional stanzas of §5, which `pip-compile` cannot emit on a supported interpreter and which must be copied back after every regeneration.

A lock is not allowed to be a partial one. `--require-hashes` demands an exact pin and a `sha256` hash for every requirement **and** for every transitive dependency, so a stanza that loses one of its hashes is an install error even where the package itself would have installed.

## 2. Regenerating a lock

Change a pin in `requirements.in` or `requirements-dev.in`, then regenerate in a separate interpreter: the versions below are the versions `pip-compile` must itself run under, not the versions the project runs under.

The regeneration environment, exactly: **CPython 3.11 with `pip==24.0` and `pip-tools==7.5.3`**. Create it as a throwaway directory that is never committed — `.lockgen/` here, or any path outside the checkout.

```bash
python3.11 -m venv .lockgen
.lockgen/bin/pip install "pip==24.0" "pip-tools==7.5.3"

.lockgen/bin/pip-compile --generate-hashes \
    --output-file=requirements.lock requirements.in

.lockgen/bin/pip-compile --allow-unsafe --generate-hashes \
    --output-file=requirements-dev.lock requirements-dev.in
```

Each `--output-file` names a committed file verbatim; a regeneration that writes to any other name is a mistake, not a variant. Resolution contacts the package index — this is the one step in the product's lifecycle that does. Running `qrcheck` never contacts anything but a loopback HTTP target.

The two `pip install` lines that bootstrap that venv are the only installs in the product's documents without `--require-hashes`, and they are exempt by necessity: they install the exact pins of the tools that generate the hashes, so no hash set for them can exist beforehand. Neither is ever installed into a project environment — an environment installed from a lock never contains the lock generator.

**Why `pip` is pinned at all.** `pip-tools` 7.5.3 imports `pip._internal.utils.compat.stdlib_pkgs`, which pip 25 removed, and fails on any newer pip with

```
ImportError: cannot import name 'stdlib_pkgs' from 'pip._internal.utils.compat'
```

So `pip==24.0` is a pin of the *regeneration environment*, not a dependency of `qrcheck`; that is why it sits in `requirements-dev.in` and therefore in the dev lock, and never in `[project].dependencies`.

**Why `--generate-hashes`.** `--require-hashes` demands an exact pin and a `sha256` hash for every requirement **and** every transitive dependency, so a lock generated without hashes cannot be installed at all, and a lock generated with only the direct pins hashed fails on the first transitive download.

**Why `--allow-unsafe` on the dev lock only.** `pip` and `setuptools` are the packages `pip-tools` marks "considered to be unsafe in a requirements file" and omits by default; without the flag the dev lock would silently drop the `pip==24.0` pin its own input declares. The runtime lock needs no such flag because it pins neither, and it must not acquire either.

**After regenerating, restore the conditional stanzas of §5** — regeneration removes them, because their markers are false for the generating interpreter — then review the diff before committing.

Verification, once the locks are in place:

```bash
python3.11 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements.lock
.venv/bin/pip install --require-hashes -r requirements-dev.lock
.venv/bin/pip install --no-deps -e .
```

`--no-deps` on the editable install is what keeps the manifest from resolving a version the lock does not name. CI runs the same three installs, as `python -m pip install --require-hashes -r requirements.lock`, then the dev lock, then `python -m pip install --no-build-isolation --no-deps -e .` (`.github/workflows/ci.yml`).

Three review rules, because a lock can be wrong while still installing:

- A regeneration that moves an unrelated transitive pin is a change to be explained, not accepted. The lock's `# via` comments record which requirement pulled each package in.
- A hash set that loses a platform wheel is a defect **even when the lock installs on the reviewer's machine**. `zxing-cpp`, `numpy` and `Pillow` each carry one `--hash` line per published wheel, and a linux-x86_64 CI job cannot notice a macOS or aarch64 wheel's hash going missing. §6 records what each platform actually needs.
- A truncated copy of a lock is not a smaller lock. The runtime lock's last stanza is `zxing-cpp`, so a copy cut short installs an environment with no primary decoder and every `qrcheck run` then exits 2 (§7); a cut inside a stanza also breaks the trailing backslash continuation, so pip rejects the file before installing anything. And a file that still names a requirement but has lost that requirement's `--hash` lines is refused outright: `--require-hashes` rejects any requirement it cannot verify, which is exactly what a truncation produces.

## 3. Runtime pins

Seven packages, registered on PyPI, pinned exactly once each:

| Distribution | Version | What `qrcheck` needs it for |
|--------------|---------|-----------------------------|
| `click` | 8.1.8 | The five subcommands, option validation, and the `CliRunner` harness the exit-code tests drive |
| `httpx` | 0.27.2 | The `http` target adapter: an explicit `httpx.Timeout` for `timeout_s`, connection reuse, and a client confined to a loopback base URL |
| `numpy` | 1.26.4 | The seeded `PCG64` generator behind every damage variant, and the array operations that measure a rendered symbol |
| `opencv-python-headless` | 4.10.0.84 | The secondary reference decoder (`cv2.QRCodeDetector`) — the second opinion of the consensus layer |
| `Pillow` | 10.4.0 | Image IO, module-grid sampling, and the resize, rotate, perspective, blur, JPEG and noise operations behind the damage cases |
| `segno` | 1.6.6 | The golden encoder, used strictly as the expectation oracle |
| `zxing-cpp` | 2.2.0 | The primary reference decoder: payload bytes, EC level, content type, and the corner position the grid sampler starts from |

Where a pin had an obvious alternative, the decision and its reason:

- **`zxing-cpp` 2.2.0.** 2.3.0 renames `Result` to `Barcode` with the same metadata surface, so it buys nothing; 3.0.0 would change the decode contract outright. 2.2.0 is also the first release that reports an EC level.
- **`opencv-python-headless`.** The headless build keeps GUI libraries out of an offline CLI. `pyzbar`/`zbar` was rejected: a system native library to install and no metadata to corroborate.
- **`segno`.** Pure Python, no dependencies on CPython 3.11, with deterministic version search and mask scoring. `qrcode` was rejected: no capacity tables, and no per-request mask control.
- **`numpy` 1.26.4.** numpy 2.x was rejected: OpenCV 4.10 predates the numpy 2 ABI, and 1.26.4 satisfies OpenCV's `numpy>=1.23.5` on CPython 3.11.
- **`httpx` over `requests`.** The adapter needs one explicit `httpx.Timeout` object mapped onto `timeout_s`, and a client that reuses a connection; `requests` offers neither as directly.
- **`click` over `argparse` and `typer`.** A subcommand group, option validation and a test harness the CLI tests already use; `typer` would add a dependency for the same result.

## 4. Development pins

Five packages, used by the test and gate tooling only:

| Distribution | Version | What the gates need it for |
|--------------|---------|----------------------------|
| `pytest` | 8.2.2 | The test runner, and the only one with native JUnit XML for the project's own CI |
| `pytest-cov` | 5.0.0 | Coverage measurement for the 85 % project floor |
| `coverage` | 7.6.1 | The engine behind `pytest-cov`, and the tool that applies the 90 % floor for `qrcheck/config`, `qrcheck/registry` and `qrcheck/report` |
| `pip-tools` | 7.5.3 | Regenerates both locks with `--generate-hashes` |
| `pip` | 24.0 | The lock-regeneration environment only, for the `pip._internal` import reason of §2 |

None of the five is a runtime dependency, and none may appear in `[project].dependencies` or in a `dev` extra: the dev lock is what installs them, and duplicating a development pin in the manifest would give the project two sources of truth for one version.

One rendering detail so no one mistakes it for drift: the dev lock carries coverage as `coverage[toml]==7.6.1`, because `pytest-cov` requires the `toml` extra and `pip-compile` merges that extra into the single coverage stanza rather than adding a second pin. On CPython 3.11 the extra adds no distribution — the standard library's `tomllib` supplies the parser — which is why the dev lock holds no companion stanza for it.

## 5. The transitive closure, and the two conditional stanzas

The closure appears **only** in the locks. Its package names are recorded here so a reader recognises them; its versions and hashes are recorded nowhere but the lock, because `--require-hashes` rejects a partially hashed file and a second copy of a version would be a second thing to keep in step.

| Lock | Packages in the closure, named but never versioned here |
|------|--------------------------------------------------------|
| `requirements.lock` | `anyio`, `certifi`, `h11`, `httpcore`, `idna`, `sniffio`, `typing-extensions` — pulled in by `httpx` and by the anyio-based transport beneath it |
| `requirements-dev.lock` | the same seven, plus `pip-tools`' own closure (`build`, `pyproject-hooks`, `wheel`, `packaging`) and `pytest`'s (`iniconfig`, `pluggy`, `packaging`), plus the unsafe block (`pip`, `setuptools`) |

Two stanzas in both locks are **conditional**: they carry an environment marker, and neither is ever installed on a supported host.

| Stanza | Marker in the lock | Pulled in by | Installed on CPython 3.11 Linux/macOS? |
|--------|--------------------|--------------|----------------------------------------|
| `colorama` | `sys_platform == "win32"` | `click` | No — Windows is out of scope |
| `importlib-metadata` | `python_version < "3.10"` | `segno` | No — the product requires 3.11 or newer |

`pip-compile` 7.5.3, like pip, drops a requirement whose marker is false for the generating interpreter, so on the only supported generating environment it cannot emit either stanza — and a regeneration of either lock removes both. They are therefore maintained by hand, at their alphabetical positions, and must be re-added after every regeneration; the lock that was in place before the regeneration remains the authority for their exact versions. Neither is installed here, and `importlib-metadata`'s own dependency `zipp` is never installed either, which is why no `zipp` stanza exists. This is the one hand-maintained exception to the rule in §1 that no lock is edited by hand.

## 6. Platform resolution matrix

The product is supported on CPython 3.11 on Linux and macOS; Windows is out of scope. This matrix is what each pin actually resolves to on those platforms. ✔ means a wheel resolves for CPython 3.11 on that platform, ✘ means none is published, and *dev* means the package is development-only and never installed by a user.

| Package | linux x86_64 | linux aarch64 | macOS arm64 | macOS x86_64 | Notes |
|---------|--------------|---------------|-------------|--------------|-------|
| `click` 8.1.8 | ✔ | ✔ | ✔ | ✔ | Pure-Python wheel, so the platform is irrelevant |
| `httpx` 0.27.2 | ✔ | ✔ | ✔ | ✔ | Pure-Python wheel |
| `numpy` 1.26.4 | ✔ | ✔ | ✔ | ✔ | A CPython 3.11 wheel per platform |
| `opencv-python-headless` 4.10.0.84 | ✔ | ✔ | ✔ | ✔ | `cp37-abi3` wheels: `manylinux_2_17_x86_64`, `manylinux_2_17_aarch64`, `macosx_11_0_arm64`, `macosx_12_0_x86_64` — the x86_64 macOS wheel needs macOS 12 or newer |
| `Pillow` 10.4.0 | ✔ | ✔ | ✔ | ✔ | A CPython 3.11 wheel per platform, `manylinux_2_28` among them |
| `segno` 1.6.6 | ✔ | ✔ | ✔ | ✔ | Pure-Python wheel, and no runtime dependency on 3.11 |
| `zxing-cpp` 2.2.0 | ✔ | **✘** | ✔ | ✔ | `manylinux_2_17_x86_64` and a `macosx_10_9_universal2` wheel covering both macOS columns; **no CPython 3.11 wheel is published for linux aarch64**, the single gap in this matrix |
| `pytest` 8.2.2 | dev | dev | dev | dev | Development only |
| `pytest-cov` 5.0.0 | dev | dev | dev | dev | Development only |
| `coverage` 7.6.1 | dev | dev | dev | dev | Development only |
| `pip-tools` 7.5.3 | dev | dev | dev | dev | Development only, and run only in the regeneration environment of §2 |
| `pip` 24.0 | dev | dev | dev | dev | Development only, and lock regeneration only |

Two notes on reading it. The `macosx_10_9_universal2` wheel serves both macOS columns, because one artefact carries both architectures; the OpenCV macOS wheel is per-architecture, which is why that row states two tags. And no environment marker papers over the one missing wheel: `requirements.in` declares none for `zxing-cpp` and `pyproject.toml` declares none either, because a marker would install an environment in which the primary decoder is absent and the authority for every assertion is gone. How that state is reported is §7.

## 7. The Linux aarch64 gap blocks the run

`zxing-cpp` 2.2.0 publishes no CPython 3.11 wheel for linux aarch64. That single gap is a **blocking failure, never a silent degradation**: a missing platform wheel is not something a run may proceed without, so the tool refuses rather than installing a fallback decoder or skipping the check that needs one.

- `qrcheck doctor` exits **2**, reporting `zxing-cpp` as `missing` and `blocking`, and its diagnostic names the distribution, the running interpreter, the platform tag and the fact that no wheel is published for that tag.
- `qrcheck run` exits **2** before any check runs, writes no `summary.json` and no evidence tree, and prints one diagnostic to standard error. Its shape on that platform:

```
the primary reference decoder zxing-cpp cannot be used for CPython 3.11.<n> on <platform tag>
because the zxingcpp module is not importable; every round-trip assertion is decoded with it,
so the run cannot start and RT-ROUNDTRIP is never silently skipped -- install zxing-cpp==2.2.0
from requirements.lock; zxing-cpp 2.2.0 publishes no wheel for <platform tag>, so the platform
itself is the obstruction rather than a broken installation
```

The platform tag is `sysconfig.get_platform()`'s, and `doctor` additionally derives the wheel tag the pin would have to match from `sys.platform` and `platform.machine()`: `manylinux_2_17_<machine>` on Linux, and `macosx_10_9_universal2` on macOS. On linux aarch64 that is `manylinux_2_17_aarch64`, the tag §6 marks ✘.

No check ever turns this into a skip: `RT-ROUNDTRIP` is the round-trip assertion every other result is read against, and a `skipped` result there would let CI report an all-passed run on a host that decoded nothing. The refusal is raised before the corpus is loaded, so the failure cannot be mistaken for a corpus or selection error.

## 8. What a missing tool costs: blocking or degrading

Each prerequisite has exactly one verdict, and the verdict decides the exit code. `qrcheck doctor` reports all of them in one pass rather than stopping at the first fault, because a reader repairing an environment needs the whole list.

| Prerequisite | Role | Pinned version | Verdict | What happens when it is not usable | Where the user sees it |
|--------------|------|----------------|---------|------------------------------------|------------------------|
| CPython interpreter | The runtime contract | 3.11 (`requires-python = ">=3.11"`) | Blocking below 3.11; degrading above | Below 3.11 the CLI exits 2 naming the running interpreter, the required version and `tomllib` as the reason; a newer 3.x or a non-CPython interpreter runs, is reported `degraded`, and still exits 0 | stderr diagnostic on start-up; the `python` row of `doctor` |
| `zxing-cpp` | Primary reference decoder — the authority for every check | 2.2.0 | **Blocking** | `doctor` and `run` exit 2 naming the distribution, the interpreter and the platform tag; no report is written; `RT-ROUNDTRIP` is never skipped | stderr diagnostic; no `summary.json` |
| `segno` | Expectation oracle | 1.6.6 | **Blocking** | Exit 2 naming the distribution and pointing at `requirements.lock`; without it no oracle-backed assertion can be made, so there is nothing to report | stderr diagnostic; the `segno` row of `doctor` |
| `numpy`, `Pillow`, `httpx`, `click` | Core libraries, imported at the point of use | As pinned in §3 | **Blocking** | Exit 2 naming the distribution and pointing at `requirements.lock`, whether it is absent or installed with a module that cannot be imported — a broken install, not a user's choice | stderr diagnostic; one `doctor` row each |
| `opencv-python-headless` | Secondary reference decoder | 4.10.0.84 | **Degrading** | The run proceeds in single-decoder mode: `agreement` records `primary_only`, and `index.html` carries a banner stating that the corroborating decoder was not installed, or was installed and could not be imported or run. When the distribution is not installed, `decoded.json`'s `per_decoder` holds the primary decoder alone; when it is installed but cannot be imported or run, `per_decoder` holds both names, the secondary at status `error` with its diagnostic in `error_text`, and no comparison was attempted, so `secondary_comparable` is `null`. No check is marked passed on the strength of a decoder that did not answer | the report artefacts, with exit 0 |
| An SVG rasterizer, or `[render] svg_rasterizer = "none"` | `RN-SVG`'s rasterization step | Not a Python package (§9) | **Degrading** | `RN-SVG` reports `skipped` with reason `missing_dependency` and records the discovery order it attempted; the skip is counted separately and never as a pass | the reason in `summary.json`, the skipped block in `index.html`, and the `svg_rasterizer` object of `doctor` |

Three properties hold across the table. A `degraded` entry never fails the gate, and a `missing` non-blocking entry never becomes a pass — degradation is recorded, never inferred as health. A `missing` entry covers two states the verdict does not distinguish, because both mean the pin cannot serve a run: the distribution is not installed, or it is installed and the module the product imports will not load, which is what a shadowing module earlier on `sys.path` or an extension built for another architecture produces. `doctor` decides a core library's status from that import rather than from `importlib.metadata`, which happily reports a version for a module that cannot load, so an installed but unusable distribution is `missing` and blocking instead of `ok`. Availability is resolved **lazily**: the probes in `qrcheck/decode/base.py` report what is installed without importing `cv2` or `zxingcpp` at module import time, so `qrcheck list`, `qrcheck loc` and `qrcheck corpus build` work on a host where a decoder is absent. And `doctor --json` emits one object with exactly six keys — `adapter`, `type`, `reachable`, `capabilities`, `dependencies`, `svg_rasterizer` — where each dependency entry carries exactly `name`, `version`, `status` and `blocking`, `status` is one of `ok`/`missing`/`degraded`, and the list is sorted by `name`. What the decode pipeline and the report do with a degradation is `architecture.md` §9.

The dependency verdicts in the table rest on one probe: `qrcheck/dependencies.py` is their single implementation, and its contract is held by two test modules — `tests/unit/test_dependencies.py` for the per-pair probe, `tests/integration/test_blocking_dependencies.py` for the blocking verdict a real `doctor` and `run` produce. It belongs with this dependency record rather than to a component of its own: it reads no configuration, decides no exit code and records no state, so `qrcheck/cli/doctor.py` turns each pair into a row of its table and `qrcheck/runner.py` turns the first unusable core dependency into the pre-flight refusal. The probe itself is standard library only and imports nothing from `qrcheck`, because it is reached when the environment is broken; it reports the distribution's installed version, read from `importlib.metadata`, together with a guarded import of the module the product really imports, so a distribution that is installed without a loadable module is unusable rather than `ok`. `CORE_DEPENDENCIES` covers `segno`, `numpy`, `Pillow`, `httpx` and `click`, while the primary decoder is probed as a prerequisite of its own and the secondary decoder's absence is a degradation rather than a refusal.

## 9. SVG rasterizers, and the order they are searched in

The `RN-SVG` check asserts that an SVG artefact rasterizes to a decodable matrix, so it needs a rasterizer — and a rasterizer is **not** a Python dependency. `[render] svg_rasterizer` accepts `auto`, `cairosvg`, `rsvg-convert`, `resvg`, `magick` or `none`; the committed default is `auto`, which probes the following order and stops at the first usable entry.

| Name | What it is | How discovery probes it | Where it comes from |
|------|------------|-------------------------|---------------------|
| `cairosvg` | A Python module that drives cairo | An import; used through its module API rather than a subprocess | Whatever environment happens to provide it — it is never installed from either lock |
| `rsvg-convert` | The command-line front end of librsvg | An executable on `PATH` | Ubuntu/Debian package `librsvg2-bin` |
| `resvg` | A standalone Rust rasterizer | An executable on `PATH` | A distribution package or a manual install |
| `magick` | ImageMagick 7 | An executable on `PATH` | Distribution package `imagemagick` |
| `convert` | ImageMagick 6 | An executable on `PATH` | Distribution package `imagemagick` |

The order is a contract rather than a preference: `doctor` and `RN-SVG` both report the sequence they attempted, so reordering it changes a report. `qrcheck/checks/render.py` holds the product's only discovery and its only rasterization; `qrcheck/cli/doctor.py` imports that discovery rather than re-implementing it, because the tool it reports has to be the tool `RN-SVG` uses. The key's validation rule and its exit-2 diagnostic are owned by `configuration.md` §6.1.

**Ubuntu, and what CI installs.** The package is `librsvg2-bin`, which provides `rsvg-convert`. It is *not* preinstalled on the `ubuntu-24.04` runner image, so the `e2e` job installs it — `sudo apt-get install -y --no-install-recommends librsvg2-bin` — and that one apt install is the only system package the pipeline adds. It is also what lets the clean-run gate require **zero skips**: without a rasterizer, `RN-SVG` would skip, and a skipped check must not be counted as coverage of SVG output.

`none` disables rasterizing altogether, and is the supported way to say so explicitly rather than by omission. With `none`, or with `auto` finding nothing, `RN-SVG` reports `skipped` with reason `missing_dependency` and records the attempted order — a skip, never a pass. Nothing is downloaded, installed or version-checked over the network at run time, and no check mutates the environment to satisfy a missing rasterizer (this is why no Python SVG rasterizer is declared as a dependency — §10).

## 10. What qrcheck deliberately does not depend on

Each absence below is a decision with a named substitute, recorded so that a convenience library does not reappear.

| Ruled out | Instead |
|-----------|---------|
| `Jinja2` or any template engine | String templating in `qrcheck/report/html.py`, which keeps `index.html` self-contained and free of a template dependency |
| `cairosvg`, `svglib` or any Python SVG rasterizer as a *declared* dependency | Runtime discovery of the rasterizer in §9, so the dependency surface stays exactly the seven pins of §3 |
| `pyzbar`/`zbar`, `zxing-cpp` 3.0 or newer, and any third decoder | The two pinned decoders; a third would change the consensus contract |
| `qrcode`, `pypng` or any second generator used by the framework itself | `segno` as the oracle only; the `reference` target is segno-backed by design, with the resulting tautology documented in `architecture.md` §9 |
| `requests`, `aiohttp`, `urllib3` used directly | `httpx` only |
| `typer`, `rich` or an `argparse` wrapper library | `click` only |
| `pandas`, `scipy`, `scikit-image` | `numpy` and `Pillow` are sufficient for every damage kind the product applies |
| `hypothesis`, `pytest-xdist`, `pytest-rerunfailures`, `flaky` | A seeded corpus and sequential execution; property-based generation, parallelism and retry machinery would each break the determinism contract |
| `ruff`, `mypy`, `black`, `isort` | No linter or type-checker is part of the product; style is PEP 8 held by review, with `python -m compileall -q qrcheck tests` as the automated syntax gate |
| Any database driver, web framework, ORM or migration tool | There is nothing to persist and no service to serve; results are files |
| Any telemetry or APM client | The reports are the observability surface, and telemetry is ruled out by the product's offline contract |
| `httpx` extras (`brotli`, `http2`, `socks`, `zstd`, `cli`) | Plain HTTP to loopback needs none of them |

**The rule this table enforces.** A new runtime dependency is a defect unless §3 already lists it. Each one would need its own entry in the platform matrix of §6, a `sha256` hash for every published wheel in the lock, a blocking-or-degrading verdict in §8 and a diagnostic in `qrcheck/cli/doctor.py` — four places a single convenience import would leave inconsistent. That is why the answer to "can we just add `pyzbar` for a third opinion" is no, and why the answer to "the package is missing on my machine" is to install from the lock rather than to add a fallback.

## 11. Where byte identity stops

One property of the pins is worth stating because a test depends on it: PNG bytes are not portable. A written PNG depends on the zlib build behind `Pillow`, so byte-identity of a damaged image is promised **only inside the pinned lock**; outside it, pixel equality is what holds. That is why the corpus and the two JSON evidence files — pure text with sorted keys and fixed separators — are byte-identical anywhere, while `damage-<kind>-<seed>.png` is byte-identical only against the versions the lock records. Artefacts a *target* produced, such as `symbol.png`, are never promised byte-identical at all, because the framework did not write them.

Timings are a different subject with a different owner: `performance.md` states the environment the latency and throughput floors are defined against, and why an absolute wall-clock figure is not portable across hosts.
