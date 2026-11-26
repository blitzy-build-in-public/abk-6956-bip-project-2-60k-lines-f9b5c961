"""The framework's single image boundary.

Every image byte the framework reads or emits passes through this module: Pillow is
imported here, and so are the open, encode, base64 and data-URI primitives, so no other
part of ``qrcheck`` needs to know how an image is opened or how it reaches a report.  The
consumers are:

* ``qrcheck.report.html`` — embeds evidence images with :func:`to_data_uri`, the only
  route an image takes into ``index.html`` (AAP §0.3.4, §0.8.4: every evidence image is
  an inline ``data:`` URI, so the report opens with no network access).
* ``qrcheck.report.evidence`` and the checks — open artefact files through
  :func:`open_image` and turn a target's ``image_bytes`` into an image through
  :func:`decode_image`.
* ``CL-ARTIFACTS`` in ``qrcheck.checks.cleanup`` — reads :func:`open_images` to attribute
  an unclosed handle to the case that leaked it (AAP §0.8.1).
* ``qrcheck.targets.reference`` — composites the logo overlay a request carries onto the
  symbol it rendered, through :func:`composite_overlay` (AAP §0.3.4: ``logo_overlay`` is a
  capability the built-in target declares, so the declaration must have behaviour).

Scope: this module *opens* and *encodes* images, and owns exactly one composite.  The
built-in target must composite the overlay it is handed and no other part of the framework
may touch an image, so :func:`composite_overlay` lives here beside :func:`open_image` and
:func:`encode_png` rather than in the target.  Every other transformation stays with the
owner it already has: resize, rotate, perspective warp, blur, JPEG re-encode and noise
synthesis belong to ``qrcheck.checks.damage``, and module-grid sampling belongs to
``qrcheck.decode.matrix``; both open their inputs through :func:`open_image`.  A second
transform API must not be added here: it would duplicate a named owner (AAP §0.7.1,
§0.8.1).

Invariants this module keeps:

* Determinism — :func:`encode_png` writes no ancillary chunk (no ``tIME``, no
  ``dpi``/``text`` metadata), so equal images encode to equal bytes and
  :func:`to_data_uri` is a pure function of its input (AAP §0.5.4).
* No policy and no I/O of its own — nothing here creates a directory, writes a file,
  reads configuration or decides whether evidence is persisted; that decision belongs to
  ``qrcheck.report.evidence``.
* No volatility — :class:`OpenImage` keeps the path exactly as given, and no hostname,
  user name, process id or timestamp is ever recorded (AAP §0.5.5).
* No adapter vocabulary — :func:`decode_image` and :func:`decode_base64` let Pillow's
  ``UnidentifiedImageError``/``OSError`` and ``binascii.Error`` propagate unchanged.  The
  adapters translate them into the framework's error types (AAP §0.3.4, §0.14.1); this
  module must not know those exist.

The only mutable module state is the registry of images that are open right now, and the
framework is single-process and sequential (AAP §0.3.6), so it needs no lock.
"""

import base64
import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# The three byte signatures this product ever sees: PNG, JPEG and SVG.  A target's bytes
# are sniffed, never trusted, so mime_type() answers application/octet-stream for
# anything else and the adapter layer decides what that means (AAP §0.3.4
# TargetContractError) — this module raises nothing for garbage.
PNG_MAGIC: bytes = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC: bytes = b"\xff\xd8\xff"
# SVG is text, so it is recognised by a leading declaration or root element after leading
# whitespace and an optional UTF-8 byte-order mark are stripped.
SVG_PREFIXES: tuple[bytes, ...] = (b"<?xml", b"<svg")

MIME_PNG: str = "image/png"
MIME_JPEG: str = "image/jpeg"
MIME_SVG: str = "image/svg+xml"
MIME_OCTET_STREAM: str = "application/octet-stream"

# mime type -> every prefix that identifies it.  A caller that sniffs SVG for itself must
# strip leading whitespace and a byte-order mark first; _looks_like_svg does exactly that,
# and mime_type() falls through to it for those forms.
MAGIC: dict[str, tuple[bytes, ...]] = {
    MIME_PNG: (PNG_MAGIC,),
    MIME_JPEG: (JPEG_MAGIC,),
    MIME_SVG: SVG_PREFIXES,
}

_UTF8_BOM: bytes = b"\xef\xbb\xbf"
_WHITESPACE: bytes = b" \t\r\n\v\f"


@dataclass(frozen=True)
class OpenImage:
    """One image that is open right now, as reported by :func:`open_images`.

    Frozen, so records compare by value: two opens of the same path for the same case
    compare equal, which is what lets ``CL-ARTIFACTS`` report a leak as a plain equality
    against an expected snapshot.

    Attributes:
        path: The path exactly as the caller gave it.  It is never resolved, because an
            absolute path must not reach a report or an artefact (AAP §0.5.5).
        case_id: The case that opened the image, or ``None`` when the open is not
            attributable to one case.
    """

    path: str
    case_id: str | None = None


# Images that are open right now, each keyed by an opaque token created per open, so
# removing an entry removes exactly the one its context created even when two identical
# paths are open at once.  Discovery order never leaks: open_images() sorts.
_OPEN_IMAGES: dict[object, OpenImage] = {}


def _looks_like_svg(data: bytes) -> bool:
    """Return whether ``data`` starts with an SVG declaration or root element.

    Strips an optional UTF-8 byte-order mark and then any leading whitespace before
    testing the prefixes in :data:`SVG_PREFIXES`, because a vendor-written SVG file may
    begin with either.
    """
    if data.startswith(_UTF8_BOM):
        data = data[len(_UTF8_BOM) :]
    return data.lstrip(_WHITESPACE).startswith(SVG_PREFIXES)


def mime_type(data: bytes) -> str:
    """Return the MIME type implied by the leading bytes of ``data``.

    Answers ``image/png``, ``image/jpeg``, ``image/svg+xml``, or
    ``application/octet-stream`` when nothing matches.  Sniffing never raises on arbitrary
    bytes: a target may return garbage, and calling that a contract failure is the
    adapter's decision (AAP §0.3.4), not this module's.

    Args:
        data: The bytes to sniff; typically a target's response body or an artefact's
            content.

    Returns:
        One of :data:`MIME_PNG`, :data:`MIME_JPEG`, :data:`MIME_SVG`,
        :data:`MIME_OCTET_STREAM`.

    Raises:
        ValueError: if ``data`` is not bytes-like.  Image data is bytes everywhere in
            the product, so any other type is a programming error rather than a finding.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"mime_type expects bytes, got {type(data).__name__}")
    head = bytes(data)
    # A signature beginning at byte 0 is answered by the table; the SVG forms that carry
    # a byte-order mark or leading whitespace need the strip _looks_like_svg performs, so
    # they fall through to it.
    for mime, prefixes in MAGIC.items():
        if head.startswith(prefixes):
            return mime
    if _looks_like_svg(head):
        return MIME_SVG
    return MIME_OCTET_STREAM


def encode_png(image: Image.Image) -> bytes:
    """Encode ``image`` as PNG bytes and return them.

    The encoding is reproducible: no ancillary chunk is written (no ``tIME``, no
    ``dpi``/``text`` metadata), so two calls on equal images return identical bytes and,
    inside the pinned Pillow/zlib build, so does a call in another process (AAP §0.5.4).
    That reproducibility is what makes a damage artefact regenerable from its seed.
    Nothing is written to disk — the image is serialised into an in-memory buffer.

    Args:
        image: The image to encode.

    Returns:
        The complete PNG file content.

    Raises:
        ValueError: if ``image`` is not a :class:`PIL.Image.Image`.
    """
    if not isinstance(image, Image.Image):
        raise ValueError(f"encode_png expects a PIL.Image.Image, got {type(image).__name__}")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def decode_image(data: bytes) -> Image.Image:
    """Open image bytes and return a fully loaded :class:`PIL.Image.Image`.

    This is how a target's ``GeneratedSymbol.image_bytes`` become an image.  The pixels
    are read eagerly, so the returned image keeps no file handle alive and the
    open-handle snapshot ``CL-ARTIFACTS`` takes stays meaningful (AAP §0.8.1).

    Pillow's own exceptions propagate unchanged: the adapter maps them to
    ``TargetContractError`` (AAP §0.14.1), and this module deliberately does not know
    that type exists.

    Args:
        data: The image file content, as received from a target or read from an artefact.

    Returns:
        The decoded image.

    Raises:
        ValueError: if ``data`` is empty or is not bytes-like.
        PIL.UnidentifiedImageError: if Pillow cannot recognise the content; it is an
            ``OSError`` subclass, so the adapter's conversion covers it.
        OSError: for truncated or otherwise broken image data.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"decode_image expects bytes, got {type(data).__name__}")
    payload = bytes(data)
    if not payload:
        raise ValueError("decode_image requires non-empty image bytes")
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


def composite_overlay(symbol: Image.Image, overlay: Image.Image) -> Image.Image:
    """Return a new RGBA image with ``overlay`` composited centred on ``symbol``.

    The one transform this module performs, and the reason it exists is a capability
    contract: ``logo_overlay`` is a member the built-in target declares, so a request that
    carries an overlay must reach a symbol carrying it -- a declared capability with no
    behaviour behind it is indistinguishable from an ignored one (AAP §0.3.4).  It lives
    here because Pillow is reachable only through this module.

    The overlay keeps its own pixel size, is centred on both axes, and is clipped when it
    is larger than the symbol in a dimension; its alpha channel is honoured, so a
    semi-transparent overlay blends with the modules under it instead of replacing them.
    Neither argument is mutated and no file is read or written: the symbol is converted
    into a fresh image first, and that copy is the canvas the overlay is pasted onto.  The
    result carries no text chunk and no timestamp, so equal inputs composite to equal
    pixels and :func:`encode_png` stays byte-deterministic (AAP §0.5.4).

    Args:
        symbol: The rendered symbol to composite onto.  Any Pillow mode is accepted; it is
            converted to ``RGBA``.
        overlay: The overlay bitmap.  It is pasted with its own alpha channel as the mask,
            so a transparent pixel leaves the symbol unchanged.

    Returns:
        A new ``RGBA`` image the size of ``symbol``.

    Raises:
        ValueError: Either argument is not a :class:`PIL.Image.Image`.  A programming
            error rather than a finding, like this module's other type refusals.
    """
    if not isinstance(symbol, Image.Image):
        raise ValueError(
            f"composite_overlay expects a PIL.Image.Image symbol, got "
            f"{type(symbol).__name__}"
        )
    if not isinstance(overlay, Image.Image):
        raise ValueError(
            f"composite_overlay expects a PIL.Image.Image overlay, got "
            f"{type(overlay).__name__}"
        )
    # ``convert`` returns a new image even when the mode already matches, so the canvas
    # below can be pasted onto without touching either argument.
    base = symbol.convert("RGBA")
    patch = overlay.convert("RGBA")
    offset = ((base.width - patch.width) // 2, (base.height - patch.height) // 2)
    # A negative offset is how Pillow clips an overlay that overhangs the symbol, so the
    # centred placement needs no crop of its own.
    base.paste(patch, offset, patch)
    return base


def encode_base64(data: bytes) -> str:
    """Return ``data`` base64-encoded as ASCII text, unwrapped.

    No newlines and no line wrapping: the data URI built by :func:`to_data_uri` and the
    corpus' ``payload_b64`` field both require the single-line form.

    Args:
        data: The bytes to encode.

    Returns:
        The base64 text, without line breaks.
    """
    return base64.b64encode(bytes(data)).decode("ascii")


def decode_base64(text: str) -> bytes:
    """Decode base64 ``text`` to bytes, rejecting malformed input.

    Decoding is strict, so a malformed body raises ``binascii.Error`` instead of quietly
    decoding a prefix of it.  The adapter converts that into ``TargetContractError``
    (AAP §0.14.1); this module lets it propagate unchanged.

    Args:
        text: The base64 text to decode.

    Returns:
        The decoded bytes; ``b""`` for an empty string.

    Raises:
        binascii.Error: if ``text`` is not valid base64 (bad padding, illegal character or
            incorrect length).
        ValueError: if ``text`` contains non-ASCII characters.
    """
    return base64.b64decode(text, validate=True)


def to_data_uri(data: bytes) -> str:
    """Return ``data`` as an inline ``data:`` URI.

    This is the only way an image reaches ``index.html``: the report embeds images
    inline, so it holds no ``link`` or ``img`` pointing at an external origin and opens
    without network access (AAP §0.3.4, §0.8.4).  The result is a pure function of
    ``data`` — no whitespace, no quoting, no origin beyond the ``data:`` scheme.

    Args:
        data: The image file content, usually :func:`encode_png` output.

    Returns:
        ``data:<mime>;base64,<payload>``, where ``<mime>`` comes from
        :func:`mime_type`.
    """
    return f"data:{mime_type(data)};base64,{encode_base64(data)}"


@contextmanager
def open_image(path: str | Path, *, case_id: str | None = None) -> Iterator[Image.Image]:
    """Open the image at ``path`` and register it as open for the life of the context.

    The image is registered before it is yielded, so :func:`open_images` reports it while
    the ``with`` body runs and no longer reports it afterwards — whether the body
    returned or raised.  ``CL-ARTIFACTS`` reads that snapshot to attribute an unclosed
    handle to the case that leaked it (AAP §0.8.1), so every image the framework touches
    is opened through here.

    The registration keeps ``path`` exactly as given and is dropped in a ``finally``
    block, together with closing the image, so a handle cannot outlive the context.

    Args:
        path: Path of the image file, as given; it is stored unresolved.
        case_id: Identifier of the case opening the image, or ``None`` when the open is
            not attributable to a case.

    Yields:
        The opened image, still holding its file handle for the caller to read.

    Raises:
        PIL.UnidentifiedImageError: if Pillow cannot recognise the file's content.
        OSError: if the file cannot be opened or read.
    """
    image = Image.open(path)
    token = object()
    _OPEN_IMAGES[token] = OpenImage(path=str(path), case_id=case_id)
    try:
        yield image
    finally:
        _OPEN_IMAGES.pop(token, None)
        image.close()


def open_images() -> tuple[OpenImage, ...]:
    """Return the images that are open right now, sorted.

    An empty tuple means nothing is open, which is what ``CL-ARTIFACTS`` asserts at the
    end of a run (AAP §0.8.1).  Entries are ordered by ``(case_id or "", path)``, so a
    report of a leak is identical on every run.

    Returns:
        The currently open images, ordered by case identifier and then by path.
    """
    return tuple(sorted(_OPEN_IMAGES.values(), key=lambda entry: (entry.case_id or "", entry.path)))


__all__ = [
    "JPEG_MAGIC",
    "MAGIC",
    "MIME_JPEG",
    "MIME_OCTET_STREAM",
    "MIME_PNG",
    "MIME_SVG",
    "OpenImage",
    "PNG_MAGIC",
    "SVG_PREFIXES",
    "composite_overlay",
    "decode_base64",
    "decode_image",
    "encode_base64",
    "encode_png",
    "mime_type",
    "open_image",
    "open_images",
    "to_data_uri",
]
