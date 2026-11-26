#!/usr/bin/env python3
"""Example QR generator driven by the ``cli`` adapter: the working counterpart of
``qrcheck/targets/cli.py``. ``tests/integration/test_cli_adapter.py`` executes it and
``docs/targets.md`` quotes the canonical table below, so that table is the one CI keeps working.

Run it from the repository root with the interpreter the dependencies are installed in::

    .venv/bin/python examples/cli_target.py --payload-b64 <BASE64> --output <PATH>

Canonical configuration:

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

Exactly one payload source is given -- ``--payload-b64 <BASE64>`` (``-`` reads stdin),
``--payload-utf8 <TEXT>``, ``--payload-stdin`` or a positional text payload -- always with
``--output <PATH>``. A template produces one argv word, so the table conveys the payload and
the artefact path alone and ``capabilities`` lists ``png`` and nothing else: declaring an
operation it cannot transmit would make the dependent check resolve as runnable. A bare
``python`` may lack ``segno``, whose exit 1 with a traceback the adapter correctly reports as
a contract error.

A request-shaped fault prints one JSON line ``{"error": "<message>"}`` to stdout and exits 2
(``TargetRejected``); a missing ``--output`` is an argparse usage error and an internal fault
exits 1 with its traceback, both contract errors. A successful run writes no stdout."""

import argparse
import base64
import binascii
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import segno

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_REJECTED = 2

EC_LEVELS = frozenset({"L", "M", "Q", "H"})
MIN_VERSION = 1
MAX_VERSION = 40
MIN_MASK = 0
MAX_MASK = 7
STDIN_SENTINEL = "-"
HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


class DeclaredRejection(Exception):
    """A request-shaped fault, reported as one JSON line plus exit code 2."""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line.

    ``--output`` is required on purpose: a missing output path is a wiring fault of
    the caller rather than a request this target may refuse, so argparse's own
    usage error (exit 2 on stderr) is the honest classification. Error level,
    format, size and colour are validated in code instead of through ``choices``,
    because a fault in them is a declared rejection, not a usage error.
    """
    parser = argparse.ArgumentParser(
        prog="cli_target.py",
        description="Generate one QR symbol as a PNG; the example 'cli' target.",
        allow_abbrev=False,
    )
    parser.add_argument("--output", required=True, metavar="PATH",
                        help="PNG path to write; its parent must already exist")
    parser.add_argument("--payload-b64", metavar="BASE64",
                        help="base64 of the payload bytes; '-' reads bytes from stdin")
    parser.add_argument("--payload-utf8", metavar="TEXT", help="payload text, as UTF-8")
    parser.add_argument("--payload-stdin", action="store_true",
                        help="read the payload bytes from stdin")
    parser.add_argument("payload", nargs="?",
                        help="payload text as UTF-8; ignored when an option names one")
    parser.add_argument("--ec", default="M", metavar="LEVEL", help="error level L|M|Q|H (M)")
    parser.add_argument("--format", default="png", metavar="FORMAT",
                        help="output format; png only")
    parser.add_argument("--size", type=int, default=8, metavar="PX",
                        help="module edge length in pixels (8)")
    parser.add_argument("--margin", type=int, default=4, metavar="MODULES",
                        help="quiet zone in modules (4)")
    parser.add_argument("--fg", default="#000000", metavar="COLOUR", help="dark module colour")
    parser.add_argument("--bg", default="#FFFFFF", metavar="COLOUR", help="light module colour")
    parser.add_argument("--version", type=int, metavar="1-40", help="version pin (none)")
    parser.add_argument("--mask", type=int, metavar="0-7", help="mask pattern pin (none)")
    return parser.parse_args(argv)


def _resolve_payload(args: argparse.Namespace) -> bytes:
    """Return the payload bytes, from exactly one source in the documented order."""
    supplied = [
        flag
        for flag, given in (
            ("--payload-b64", args.payload_b64 is not None),
            ("--payload-utf8", args.payload_utf8 is not None),
            ("--payload-stdin", args.payload_stdin),
        )
        if given
    ]
    if len(supplied) > 1:
        raise DeclaredRejection("exactly one payload source may be given")
    if args.payload_b64 is not None:
        if args.payload_b64 == STDIN_SENTINEL:
            return sys.stdin.buffer.read()
        try:
            return base64.b64decode(args.payload_b64, validate=True)
        except (binascii.Error, ValueError):
            raise DeclaredRejection("invalid base64 payload") from None
    if args.payload_utf8 is not None:
        return args.payload_utf8.encode("utf-8")
    if args.payload_stdin:
        return sys.stdin.buffer.read()
    if args.payload is not None:
        return args.payload.encode("utf-8")
    raise DeclaredRejection(
        "no payload supplied: pass --payload-b64, --payload-utf8 or --payload-stdin"
    )


def _resolve_ec(value: str) -> str:
    """Return the error level, rejecting anything outside L, M, Q and H."""
    level = value.strip().upper()
    if level not in EC_LEVELS:
        raise DeclaredRejection(f"invalid error correction level '{value}'")
    return level


def _resolve_format(value: str) -> str:
    """Return the output format, rejecting everything this script cannot emit."""
    if value.strip().lower() != "png":
        raise DeclaredRejection(
            f"unsupported output format '{value}': examples/cli_target.py emits png only"
        )
    return "png"


def _resolve_size(value: int) -> int:
    """Return the module size in pixels, which must be positive."""
    if value <= 0:
        raise DeclaredRejection("module size must be greater than 0")
    return value


def _resolve_margin(value: int) -> int:
    """Return the quiet zone in modules, which must not be negative."""
    if value < 0:
        raise DeclaredRejection("quiet zone must not be negative")
    return value


def _resolve_colour(value: str) -> str:
    """Return the colour as uppercase ``#RRGGBB``, expanding a ``#RGB`` shorthand."""
    digits = value[1:] if value.startswith("#") else ""
    if len(digits) == 3:
        digits = "".join(digit * 2 for digit in digits)
    if len(digits) != 6 or not set(digits) <= HEX_DIGITS:
        raise DeclaredRejection(f"invalid colour '{value}': expected #RRGGBB or #RGB")
    return "#" + digits.upper()


def _resolve_version(value: int | None) -> int | None:
    """Return the version pin unchanged, or reject it when outside 1-40."""
    if value is not None and not MIN_VERSION <= value <= MAX_VERSION:
        raise DeclaredRejection("version must be between 1 and 40")
    return value


def _resolve_mask(value: int | None) -> int | None:
    """Return the mask pin unchanged, or reject it when outside 0-7."""
    if value is not None and not MIN_MASK <= value <= MAX_MASK:
        raise DeclaredRejection("mask must be between 0 and 7")
    return value


def _encode(payload: bytes, ec: str, version: int | None, mask: int | None) -> segno.QRCode:
    """Encode ``payload`` as a Model 2 QR symbol at the requested error level."""
    # Both defaults of segno are wrong here, so both are stated. make_qr is
    # Model 2 by construction, where bare segno.make() returns a Micro QR symbol for
    # any payload that fits one -- and a Micro symbol has no 4V+17 module count, no
    # Version 1-40 number and a different format-information layout.
    # boost_error=False keeps the requested error level instead of spending spare
    # capacity to raise it at the same version, which would make the symbol carry a
    # level the caller never asked for.
    try:
        return segno.make_qr(payload, error=ec, version=version, mask=mask, boost_error=False)
    except segno.DataOverflowError:
        # Caught before ValueError: DataOverflowError is a ValueError subclass.
        pinned = "" if version is None else f" for version {version}"
        raise DeclaredRejection(
            f"data too large{pinned} at error level {ec}: {len(payload)} bytes requested"
        ) from None
    except (ValueError, TypeError) as exc:
        # segno's own message names the offending parameter and the fault is still
        # request-shaped, so it is relayed rather than paraphrased.
        raise DeclaredRejection(str(exc) or "segno rejected the request") from None


def _render(qr: segno.QRCode, output: Path, output_format: str, size: int, margin: int,
            foreground: str, background: str) -> None:
    """Write the encoded symbol to ``output``; no resize, crop or re-encode."""
    # kind is explicit and never inferred from the file extension, so the artefact is
    # segno's own PNG and is (4V + 17 + 2*margin) * size pixels wide by construction.
    qr.save(str(output), kind=output_format, scale=size, border=margin,
            dark=foreground, light=background)


def _emit_rejection(message: str) -> int:
    """Write the single JSON rejection line and return the rejection exit code."""
    sys.stdout.write(json.dumps({"error": message}) + "\n")
    sys.stdout.flush()
    return EXIT_REJECTED


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one symbol: 0 when written, 2 when refused, 1 on an internal fault."""
    args = _parse_args(argv)
    try:
        payload = _resolve_payload(args)
        if not payload:
            raise DeclaredRejection("empty payload: qrcheck requires at least one byte")
        ec = _resolve_ec(args.ec)
        output_format = _resolve_format(args.format)
        size = _resolve_size(args.size)
        margin = _resolve_margin(args.margin)
        foreground = _resolve_colour(args.fg)
        background = _resolve_colour(args.bg)
        version = _resolve_version(args.version)
        mask = _resolve_mask(args.mask)
        qr = _encode(payload, ec, version, mask)
        _render(qr, Path(args.output), output_format, size, margin, foreground, background)
    except DeclaredRejection as rejection:
        return _emit_rejection(str(rejection))
    except OSError:
        # An unwritable output path is an environment fault, not a refusal, so the
        # traceback goes to stderr and the adapter reports a contract error.
        traceback.print_exc(file=sys.stderr)
        return EXIT_INTERNAL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
