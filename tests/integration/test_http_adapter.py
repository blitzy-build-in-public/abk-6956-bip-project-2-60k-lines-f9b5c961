"""The ``http`` adapter over its real transport: a loopback generator, driven by configuration.

AAP §0.3.3 gives the ``http`` target one contract: the vendor shape lives entirely in
configuration -- ``url``, ``method``, ``request_template``, ``image_field``,
``image_encoding`` and ``metadata_map`` -- and the adapter knows none of it,
``tests/unit/test_targets_transport.py`` holding the mechanical half as an AST scan. This
module holds the positive half over a real socket, driving ``examples/http_target_server.py``
as a real process and measuring every assertion from the artefact it answered with.

A non-2xx status is :class:`TargetContractError`, not ``TargetRejected``, because an HTTP
target has no declared-rejection channel (AAP §0.14.1), and so is a configured
``metadata_map`` path the response does not carry -- while a field that is not configured
yields ``None``. Every URL is the fixture's own loopback base, every measurement is a pixel
value or a decoded field, and nothing is skipped, marked, retried or mocked (AAP §0.4.3,
§0.5.6)."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, Protocol

import httpx
import pytest
from PIL import Image, ImageChops

from qrcheck.config.load import load_config
from qrcheck.corpus.capacity import minimal_version
from qrcheck.decode import decode_symbol
from qrcheck.targets import ADAPTER_TYPES
from qrcheck.targets.base import (
    Adapter,
    GenerateRequest,
    GeneratedSymbol,
    TargetContractError,
    TargetUnavailable,
    TargetUnsupported,
    build_request,
)

# The example server's routes: the example's own shape, configured into a table, not the adapter's.

_GENERATE_ROUTE: Final[str] = "/generate"
_RAW_ROUTE: Final[str] = "/raw"

#: The eight-byte signature every PNG begins with: an artefact-format assertion with no library.
_PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"

# The fixed inputs: each chosen so a transport that dropped the value could not match the artefact.

#: The payload every case requests; it carries a non-ASCII character, so lossy UTF-8 shows.
_PAYLOAD_TEXT: Final[str] = "qrcheck http adapter integration payload \u2713"
_PAYLOAD: Final[bytes] = _PAYLOAD_TEXT.encode("utf-8")

#: A payload that is not valid UTF-8: only the base64 channel can carry it, so the two differ.
_BINARY_PAYLOAD: Final[bytes] = bytes(range(0x80, 0x88))

#: The dark-module colour: no channel is 0 or 255, so a target drawing black could not match it.
_FOREGROUND: Final[tuple[int, int, int]] = (0x12, 0x34, 0x56)

#: The light-module colour, likewise off-white rather than white.
_BACKGROUND: Final[tuple[int, int, int]] = (0xDD, 0xEE, 0xFF)

#: Module edge and quiet zone for the geometry cases: unequal, and neither the example's default.
_MODULE_PX: Final[int] = 6
_QUIET_ZONE_MODULES: Final[int] = 2

#: The requested levels: the fixtures' one plus a second, so ``{ec_level}`` is proved transmitted.
_EC_LEVEL: Final[str] = "M"
_OTHER_EC_LEVEL: Final[str] = "Q"

#: The capability names the example's routes honour: ``png`` plus what the placeholders convey.
_TEMPLATE_CAPABILITIES: Final[tuple[str, ...]] = (
    "png",
    "ec_level",
    "module_size",
    "quiet_zone",
    "color",
    "metadata_version",
    "metadata_ec",
    "metadata_mask",
    "metadata_mode",
)

#: The envelope route's request body, the table the example's own docstring quotes.
_REQUEST_TEMPLATE: Final[Mapping[str, str]] = {
    "payload_b64": "{payload_b64}",
    "ec_level": "{ec_level}",
    "format": "{format}",
    "module_size": "{size}",
    "quiet_zone": "{margin}",
    "foreground": "{fg}",
    "background": "{bg}",
}

#: The same body with the text payload channel instead of the base64 one.
_TEXT_REQUEST_TEMPLATE: Final[Mapping[str, str]] = {
    "payload_utf8": "{payload_utf8}",
    "ec_level": "{ec_level}",
    "format": "{format}",
}

#: The bounded run's request body: ``RT-ROUNDTRIP`` pins no geometry, colour, version or mask.
_BOUNDED_TEMPLATE: Final[Mapping[str, str]] = {
    "payload_b64": "{payload_b64}",
    "ec_level": "{ec_level}",
    "format": "{format}",
}

#: Framework field -> dotted path into the response; the left-hand names are the product's own.
_METADATA_MAP: Final[Mapping[str, str]] = {
    "version": "metadata.version",
    "ec": "metadata.ec_level",
    "mask": "metadata.mask",
    "mode": "metadata.mode",
}

#: The corpus class whose records the bounded run drives, with the count AAP §0.8.2 declares.
_BOUNDED_CLASS: Final[str] = "otp"
_BOUNDED_CLASS_RECORDS: Final[int] = 12

#: The one check the bounded run selects: it needs nothing but ``png``.
_BOUNDED_CHECK: Final[str] = "RT-ROUNDTRIP"

#: What a target that declares no metadata must report, stated once so test 6 stays one rule.
_NO_DECLARED_METADATA: Final[Mapping[str, None]] = {
    "declared_version": None,
    "declared_ec": None,
    "declared_mask": None,
    "declared_mode": None,
}

#: The two requests the example refuses, each with the wire field it varies; both are built outside
#: ``build_request``, since the point is an illegal request reaching a target.
_REJECTED_REQUESTS: Final[tuple[Any, ...]] = (
    pytest.param({"ec_level": "X"}, {"ec_level": "X"}, id="ec-level-outside-the-enum"),
    pytest.param({"module_size_px": 0}, {"module_size": 0}, id="module-size-zero"),
)


# The two fixture contracts this module borrows, declared as protocols because a conftest is not an
# importable module of the package under test; stating the members keeps the coupling visible.


class _ServerCarrier(Protocol):
    """What the ``http_target_server`` fixture hands out: an address, and how to stop it."""

    @property
    def port(self) -> int:
        """The ephemeral port the example server bound."""

    @property
    def base_url(self) -> str:
        """The server's loopback base URL, ``http://127.0.0.1:<port>``."""

    def stop(self) -> None:
        """Stop the server; idempotent."""


class _CliOutcome(Protocol):
    """What the ``run_cli`` fixture returns: a status and both captured streams."""

    exit_code: int
    stdout: str
    stderr: str


# Configuration and adapter lifecycle


def _http_target(
    server: _ServerCarrier,
    *,
    route: str,
    template: Mapping[str, str],
    capabilities: Sequence[str],
    image_field: str = "image_b64",
    image_encoding: str = "base64",
    metadata_map: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return one ``[targets.<name>]`` table for the running example server."""
    table: dict[str, object] = {
        "type": "http",
        "url": f"{server.base_url}{route}",
        "method": "POST",
        "image_field": image_field,
        "image_encoding": image_encoding,
        "capabilities": list(capabilities),
        "request_template": dict(template),
    }
    if metadata_map is not None:
        table["metadata_map"] = dict(metadata_map)
    return table


def _write_config(
    write: Callable[..., Path],
    tmp_path: Path,
    targets: Mapping[str, Mapping[str, object]],
    *,
    default_adapter: str | None = None,
) -> Path:
    """Write a schema-valid document under ``tmp_path`` and return its path."""
    return write(
        tmp_path / "qrcheck.toml",
        default_adapter=default_adapter or sorted(targets)[0],
        report_dir=tmp_path / "reports",
        targets=targets,
    )


@contextmanager
def _open_adapter(config_path: Path, name: str) -> Iterator[Adapter]:
    """Load ``config_path``, build the target called ``name``, and close it afterwards."""
    config = load_config(config_path)
    target = config.targets[name]
    adapter = ADAPTER_TYPES[target.type](target, name=name)
    try:
        yield adapter
    finally:
        adapter.close()


# Measuring the artefact: the proofs measure pixels rather than echo the request, because a quiet
# repeating it back would prove only that the framework can serialise a dict, while a quiet zone, a
# module colour and a pixel dimension are properties of the symbol actually produced.


def _png_image(symbol: GeneratedSymbol) -> Image.Image:
    """Return the symbol's artefact as an RGB image, refusing anything that is not a PNG."""
    payload = symbol.image_bytes
    assert payload.startswith(_PNG_SIGNATURE), (
        f"the target answered {payload[:8]!r}, which does not begin with the PNG "
        f"signature {_PNG_SIGNATURE!r}"
    )
    with Image.open(io.BytesIO(payload)) as opened:
        return opened.convert("RGB")


def _border_pixels(
    image: Image.Image,
    background: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Return the all-background border on each side, as ``(left, top, right, bottom)`` pixels."""
    flat = Image.new("RGB", image.size, background)
    difference = ImageChops.difference(image, flat).convert("L")
    mask = difference.point(lambda value: 255 if value else 0)
    box = mask.getbbox()
    assert box is not None, (
        "the artefact holds no module at all: every pixel equals the background colour "
        f"{background!r}"
    )
    left, upper, right, lower = box
    width, height = image.size
    return (left, upper, width - right, height - lower)


def _module_centre(index: int, *, margin: int, module_px: int) -> int:
    """Return the pixel coordinate half a module into module ``index`` along one axis."""
    return (margin + index) * module_px + module_px // 2


def _module_count(version: int) -> int:
    """Return the side length in modules of a Model 2 symbol of ``version``: ``N = 4V + 17``."""
    return 4 * version + 17


def _out_of_contract(**overrides: Any) -> GenerateRequest:
    """Build a request the contract forbids, unvalidated by construction."""
    return GenerateRequest(payload=_PAYLOAD, **overrides)


# 1 and 2: the two ways an artefact can arrive


def test_base64_image_field_decodes_to_the_requested_payload(
    http_target_server: _ServerCarrier,
    write_config: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A base64 value at a configured dotted path decodes to exactly the requested payload."""
    config = _write_config(
        write_config,
        tmp_path,
        {
            "envelope": _http_target(
                http_target_server,
                route=_GENERATE_ROUTE,
                template=_REQUEST_TEMPLATE,
                capabilities=_TEMPLATE_CAPABILITIES,
                metadata_map=_METADATA_MAP,
            )
        },
    )
    request = build_request(_PAYLOAD, _EC_LEVEL)

    with _open_adapter(config, "envelope") as adapter:
        symbol = adapter.generate(request)

    # The artefact: a real PNG, of the format the request asked for.
    assert symbol.image_bytes.startswith(_PNG_SIGNATURE), (
        f"the target answered {symbol.image_bytes[:8]!r}, not a PNG"
    )
    assert symbol.output_format == "png"

    # The metadata through the configured map: ``declared_version`` vs the oracle's version.
    assert symbol.declared_version == minimal_version(len(_PAYLOAD), _EC_LEVEL, "byte"), (
        f"declared version {symbol.declared_version!r} is not the minimal version for "
        f"{len(_PAYLOAD)} bytes at EC {_EC_LEVEL}"
    )
    assert symbol.declared_ec == _EC_LEVEL
    assert symbol.declared_mask in range(8), (
        f"declared mask {symbol.declared_mask!r} is outside the eight a Model 2 symbol "
        "may use"
    )
    assert symbol.declared_mode in {"numeric", "alphanumeric", "byte"}

    # The payload: the bytes the primary decoder returns are the bytes that were asked for.
    decoded = decode_symbol(symbol.image_bytes, request)
    assert decoded.payload_bytes == _PAYLOAD, (
        f"decoded payload {decoded.payload_bytes!r} differs from the requested "
        f"{_PAYLOAD!r} (agreement {decoded.agreement!r})"
    )


def test_raw_body_image_field_decodes_to_the_requested_payload(
    http_target_server: _ServerCarrier,
    write_config: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``image_field = "_body"`` with ``image_encoding = "raw"`` decodes the same symbol."""
    config = _write_config(
        write_config,
        tmp_path,
        {
            "raw_body": _http_target(
                http_target_server,
                route=_RAW_ROUTE,
                # The raw route answers with the body itself, so payload and format suffice.
                template={"payload_b64": "{payload_b64}", "format": "{format}"},
                capabilities=("png",),
                image_field="_body",
                image_encoding="raw",
            )
        },
    )
    request = build_request(_PAYLOAD, _EC_LEVEL)

    with _open_adapter(config, "raw_body") as adapter:
        symbol = adapter.generate(request)

    assert symbol.image_bytes.startswith(_PNG_SIGNATURE), (
        f"the target answered {symbol.image_bytes[:8]!r}, not the raw PNG body"
    )
    assert symbol.output_format == "png"
    for field_name, expected in sorted(_NO_DECLARED_METADATA.items()):
        assert getattr(symbol, field_name) is expected, (
            f"{field_name} is {getattr(symbol, field_name)!r} for a target that declares "
            "no metadata; a raw body has no envelope to read a claim from"
        )

    decoded = decode_symbol(symbol.image_bytes, request)
    assert decoded.payload_bytes == _PAYLOAD, (
        f"decoded payload {decoded.payload_bytes!r} differs from the requested "
        f"{_PAYLOAD!r}"
    )


# 3: all eight placeholders, each proved by its effect on the artefact


def test_every_template_placeholder_takes_effect(
    http_target_server: _ServerCarrier,
    write_config: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Every one of the eight placeholders changes the symbol the server returns."""
    config = _write_config(
        write_config,
        tmp_path,
        {
            "envelope": _http_target(
                http_target_server,
                route=_GENERATE_ROUTE,
                template=_REQUEST_TEMPLATE,
                # ``svg`` is declared so an ``svg`` request travels rather than being refused first.
                capabilities=(*_TEMPLATE_CAPABILITIES, "svg"),
                metadata_map=_METADATA_MAP,
            ),
            "text": _http_target(
                http_target_server,
                route=_GENERATE_ROUTE,
                template=_TEXT_REQUEST_TEMPLATE,
                capabilities=("png", "ec_level"),
            ),
        },
    )

    with (
        _open_adapter(config, "envelope") as envelope,
        _open_adapter(config, "text") as text,
    ):
        # -- {payload_b64}: the binary payload can travel no other way ------------------
        binary_request = build_request(_BINARY_PAYLOAD, _EC_LEVEL)
        binary_symbol = envelope.generate(binary_request)
        assert decode_symbol(binary_symbol.image_bytes, binary_request).payload_bytes == (
            _BINARY_PAYLOAD
        ), "the base64 channel did not carry the payload's bytes unchanged"

        # -- {payload_utf8}: the same payload through the text channel ------------------
        text_request = build_request(_PAYLOAD, _EC_LEVEL)
        text_symbol = text.generate(text_request)
        assert decode_symbol(text_symbol.image_bytes, text_request).payload_bytes == _PAYLOAD
        with pytest.raises(TargetUnsupported):
            # A payload that is not valid UTF-8 has no faithful text form, so it is refused.
            text.generate(build_request(_BINARY_PAYLOAD, _EC_LEVEL))

        # -- {ec_level}: the target's claim and both independent readings ---------------
        ec_request = build_request(_PAYLOAD, _OTHER_EC_LEVEL)
        ec_symbol = envelope.generate(ec_request)
        ec_decoded = decode_symbol(ec_symbol.image_bytes, ec_request)
        assert ec_symbol.declared_ec == _OTHER_EC_LEVEL, (
            f"the target declares EC {ec_symbol.declared_ec!r} for a request that asked "
            f"for {_OTHER_EC_LEVEL!r}"
        )
        assert ec_decoded.ec_level_decoder == _OTHER_EC_LEVEL, (
            f"the primary decoder read EC {ec_decoded.ec_level_decoder!r} from a symbol "
            f"requested at {_OTHER_EC_LEVEL!r}"
        )
        assert ec_decoded.ec_level_format == _OTHER_EC_LEVEL, (
            f"the format information carries EC {ec_decoded.ec_level_format!r} for a "
            f"symbol requested at {_OTHER_EC_LEVEL!r}"
        )

        # -- {format}: a PNG artefact, then the example's rejection of any other --------
        png_request = build_request(_PAYLOAD, _EC_LEVEL)
        png_symbol = envelope.generate(png_request)
        with Image.open(io.BytesIO(png_symbol.image_bytes)) as opened:
            assert opened.format == "PNG", (
                f"a png request produced an artefact the image library reads as "
                f"{opened.format!r}"
            )
        svg_request = build_request(_PAYLOAD, _EC_LEVEL, output_format="svg")
        with pytest.raises(TargetContractError) as refused:
            envelope.generate(svg_request)
        assert "unsupported output format" in str(refused.value), (
            "a request asking for svg did not reach the server's format check, so the "
            f"format placeholder was not transmitted: {str(refused.value)!r}"
        )

        # -- {size}: the dimensions follow (N + 2 * margin) * size ---------------------
        for module_px in (_MODULE_PX, _MODULE_PX + 3):
            size_request = build_request(
                _PAYLOAD,
                _EC_LEVEL,
                module_size_px=module_px,
                quiet_zone_modules=_QUIET_ZONE_MODULES,
            )
            size_symbol = envelope.generate(size_request)
            size_image = _png_image(size_symbol)
            version = decode_symbol(size_symbol.image_bytes, size_request).version
            assert version is not None, "the sampled grid gave no version"
            expected_px = (_module_count(version) + 2 * _QUIET_ZONE_MODULES) * module_px
            assert size_image.size == (expected_px, expected_px), (
                f"module size {module_px} produced {size_image.size!r}; a Version "
                f"{version} symbol with a {_QUIET_ZONE_MODULES}-module quiet zone is "
                f"{expected_px}x{expected_px} px"
            )

        # -- {margin}: the measured border is margin * size on every side ---------------
        measured: dict[int, tuple[int, int, int, int]] = {}
        for margin in (_QUIET_ZONE_MODULES, _QUIET_ZONE_MODULES + 2):
            margin_request = build_request(
                _PAYLOAD,
                _EC_LEVEL,
                module_size_px=_MODULE_PX,
                quiet_zone_modules=margin,
                foreground_rgb=_FOREGROUND,
                background_rgb=_BACKGROUND,
            )
            margin_image = _png_image(envelope.generate(margin_request))
            border = _border_pixels(margin_image, _BACKGROUND)
            expected_border = (margin * _MODULE_PX,) * 4
            assert border == expected_border, (
                f"a {margin}-module quiet zone at {_MODULE_PX} px/module measured "
                f"{border!r} (left, top, right, bottom) rather than {expected_border!r}"
            )
            measured[margin] = border
        # Two modules more per side at the same size is two modules of pixels: the value travelled.
        narrow = measured[_QUIET_ZONE_MODULES]
        wide = measured[_QUIET_ZONE_MODULES + 2]
        growth = [later - earlier for earlier, later in zip(narrow, wide)]
        assert growth == [2 * _MODULE_PX] * 4, (
            f"a two-module larger quiet zone changed the border from {narrow!r} to "
            f"{wide!r}; each side should have grown by {2 * _MODULE_PX} px"
        )

        # -- {fg} and {bg}: the sampled pixels are the requested colours ---------------
        sample_image = _png_image(
            envelope.generate(
                build_request(
                    _PAYLOAD,
                    _EC_LEVEL,
                    module_size_px=_MODULE_PX,
                    quiet_zone_modules=_QUIET_ZONE_MODULES,
                    foreground_rgb=_FOREGROUND,
                    background_rgb=_BACKGROUND,
                )
            )
        )
        pixels = sample_image.load()
        # Module (2, 2) is the centre of the top-left finder pattern's dark square, so it is dark.
        centre = _module_centre(2, margin=_QUIET_ZONE_MODULES, module_px=_MODULE_PX)
        assert pixels[centre, centre] == _FOREGROUND, (
            f"the dark module at ({centre}, {centre}) is {pixels[centre, centre]!r}, not "
            f"the requested foreground {_FOREGROUND!r}"
        )
        assert pixels[0, 0] == _BACKGROUND, (
            f"the quiet zone's first pixel is {pixels[0, 0]!r}, not the requested "
            f"background {_BACKGROUND!r}"
        )


# 4: a rejected request keeps the target's own words


@pytest.mark.parametrize(("overrides", "wire_fields"), _REJECTED_REQUESTS)
def test_an_error_response_becomes_a_contract_error_with_the_text_preserved(
    overrides: Mapping[str, Any],
    wire_fields: Mapping[str, Any],
    http_target_server: _ServerCarrier,
    write_config: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A rejected request's error text reaches both the diagnostic and the evidence."""
    config = _write_config(
        write_config,
        tmp_path,
        {
            "envelope": _http_target(
                http_target_server,
                route=_GENERATE_ROUTE,
                template=_REQUEST_TEMPLATE,
                capabilities=_TEMPLATE_CAPABILITIES,
                metadata_map=_METADATA_MAP,
            )
        },
    )
    request = _out_of_contract(**overrides)
    wire_body: dict[str, Any] = {
        "payload_b64": request.as_dict()["payload_b64"],
        "format": request.output_format,
        **wire_fields,
    }
    response = httpx.post(
        f"{http_target_server.base_url}{_GENERATE_ROUTE}",
        json=wire_body,
        # A harness guard against a hung loopback server, not a product threshold.
        timeout=30.0,
    )
    assert response.status_code == 400, (
        f"the example server answered {response.status_code} to a request it is meant to "
        f"reject, so there is no error text to compare against: {response.text!r}"
    )
    declared = response.json()["error"]
    assert isinstance(declared, str) and declared, (
        f"the server's 400 carried no error text: {response.text!r}"
    )

    with _open_adapter(config, "envelope") as adapter:
        with pytest.raises(TargetContractError) as raised:
            adapter.generate(request)

    # The diagnostic quotes the body as a ``repr``, so a quote arrives escaped; undo exactly that.
    diagnostic = str(raised.value).replace("\\'", "'")
    assert declared in diagnostic, (
        f"the target's own error text {declared!r} is missing from the diagnostic "
        f"{diagnostic!r}"
    )
    assert declared.encode("utf-8") in raised.value.body_prefix, (
        f"the response body the target refused with is not carried as evidence; "
        f"body_prefix holds {raised.value.body_prefix!r}"
    )


# 5: a transport that cannot connect is reported, not retried


def test_a_stopped_server_is_unavailable(
    http_target_server: _ServerCarrier,
    write_config: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A refused connection is ``TargetUnavailable``, and the stop is what makes it so."""
    config = _write_config(
        write_config,
        tmp_path,
        {
            "envelope": _http_target(
                http_target_server,
                route=_GENERATE_ROUTE,
                template=_REQUEST_TEMPLATE,
                capabilities=_TEMPLATE_CAPABILITIES,
                metadata_map=_METADATA_MAP,
            )
        },
    )
    request = build_request(_PAYLOAD, _EC_LEVEL)

    with _open_adapter(config, "envelope") as adapter:
        # Sensitivity: while the server is up, this exact call succeeds.
        running = adapter.generate(request)
        assert running.image_bytes.startswith(_PNG_SIGNATURE), (
            "the same request did not succeed before the server was stopped, so the "
            "unavailability assertion below would prove nothing"
        )

        http_target_server.stop()

        with pytest.raises(TargetUnavailable):
            adapter.generate(request)


# 6: metadata that is absent, and metadata that was never configured


def test_an_absent_configured_metadata_path_is_a_contract_error(
    http_target_server: _ServerCarrier,
    write_config: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Configured metadata the response lacks is a contract error; unconfigured metadata is None."""
    config = _write_config(
        write_config,
        tmp_path,
        {
            "absent_path": _http_target(
                http_target_server,
                route=_GENERATE_ROUTE,
                template=_REQUEST_TEMPLATE,
                capabilities=_TEMPLATE_CAPABILITIES,
                metadata_map={"version": "metadata.no_such_field"},
            ),
            "partial_metadata": _http_target(
                http_target_server,
                route=_GENERATE_ROUTE,
                template=_REQUEST_TEMPLATE,
                capabilities=_TEMPLATE_CAPABILITIES,
                # Only two of the four fields are configured: the other two must arrive as None.
                metadata_map={"mask": "metadata.mask", "mode": "metadata.mode"},
            ),
        },
    )
    request = build_request(_PAYLOAD, _EC_LEVEL)

    with _open_adapter(config, "partial_metadata") as partial:
        symbol = partial.generate(request)
    assert symbol.declared_version is None, (
        f"declared_version is {symbol.declared_version!r} for a field no metadata path "
        "was configured for"
    )
    assert symbol.declared_ec is None, (
        f"declared_ec is {symbol.declared_ec!r} for a field no metadata path was "
        "configured for"
    )
    assert symbol.declared_mask in range(8), (
        f"declared_mask is {symbol.declared_mask!r} although its configured path carries it"
    )
    assert symbol.declared_mode, (
        f"declared_mode is {symbol.declared_mode!r} although its configured path carries it"
    )

    with _open_adapter(config, "absent_path") as broken:
        with pytest.raises(TargetContractError) as raised:
            broken.generate(request)
    message = str(raised.value)
    assert "version" in message, (
        f"the diagnostic does not name the framework field whose path is absent: {message!r}"
    )
    assert "metadata.no_such_field" in message, (
        f"the diagnostic does not name the configured path that the response lacks: "
        f"{message!r}"
    )


# 7: the whole product over the same transport


def test_a_bounded_run_through_the_http_adapter_passes(
    http_target_server: _ServerCarrier,
    write_config: Callable[..., Path],
    run_cli: Callable[..., _CliOutcome],
    tmp_path: Path,
) -> None:
    """``run --checks RT-ROUNDTRIP --class otp`` drives the loopback target and exits 0."""
    report_dir = tmp_path / "reports"
    config = _write_config(
        write_config,
        tmp_path,
        {
            "loopback": _http_target(
                http_target_server,
                route=_GENERATE_ROUTE,
                template=_BOUNDED_TEMPLATE,
                capabilities=("png",),
            )
        },
        default_adapter="loopback",
    )

    outcome = run_cli(
        "run",
        "--adapter",
        "loopback",
        "--ec",
        _EC_LEVEL,
        "--checks",
        _BOUNDED_CHECK,
        "--class",
        _BOUNDED_CLASS,
        config=config,
    )

    assert outcome.exit_code == 0, (
        f"the bounded run exited {outcome.exit_code}; stdout {outcome.stdout!r}, "
        f"stderr {outcome.stderr!r}"
    )
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"] == {
        "passed": _BOUNDED_CLASS_RECORDS,
        "failed": 0,
        "skipped": 0,
        "error": 0,
    }, f"the bounded run counted {summary['counts']!r}"
    results = summary["results"]
    assert len(results) == _BOUNDED_CLASS_RECORDS, (
        f"the run reports {len(results)} results for {_BOUNDED_CLASS_RECORDS} records"
    )
    assert {result["check_id"] for result in results} == {_BOUNDED_CHECK}
    assert {result["status"] for result in results} == {"passed"}, (
        "a result other than passed came back from a conforming loopback target: "
        f"{sorted({result['status'] for result in results})}"
    )
