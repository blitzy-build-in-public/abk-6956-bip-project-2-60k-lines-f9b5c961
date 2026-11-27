"""The framework's single image boundary.

Every image byte the framework reads or emits passes through this module: Pillow is imported here,
and so are the open, encode, base64 and data-URI primitives, so no other part of ``qrcheck`` needs
to know how an image is opened or how it reaches a report. ``qrcheck.report.html`` embeds evidence
images with :func:`to_data_uri`, the only route an image takes into ``index.html``: every evidence
image is an inline ``data:`` URI, so the report opens with no network access, and
``qrcheck.targets.reference`` composites the logo overlay a request carries through
:func:`composite_overlay`, because ``logo_overlay`` is a capability the built-in target declares and
a declared capability must have behaviour behind it.

Scope: this module *opens* and *encodes* images and owns exactly one composite, because Pillow is
reachable only through here. Every other transformation stays with the owner it already has --
resize, rotate, perspective warp, blur, JPEG re-encode and noise synthesis in
``qrcheck.checks.damage``, module-grid sampling in ``qrcheck.decode.matrix`` -- and a second
transform API must not be added here.

Invariants: determinism, since :func:`encode_png` writes no ancillary chunk (no ``tIME``, no
``dpi``/``text`` metadata), so equal images encode to equal bytes and :func:`to_data_uri` is a pure
function; no policy and no I/O of its own, because whether evidence is persisted belongs to
``qrcheck.report.evidence``; no volatility, since :class:`OpenImage` keeps the path exactly as
given and no hostname, user name, process id or timestamp is recorded; and no adapter vocabulary,
since :func:`decode_image` and :func:`decode_base64` let Pillow's
``UnidentifiedImageError``/``OSError`` and ``binascii.Error`` propagate unchanged for the adapters
to translate. The only mutable module state is the registry of images open right now, and the
framework is single-process and sequential, so it needs no lock.
"""

import base64
import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# ``Image`` is a declared name of this module's public surface rather than an incidental import:
# the name is Pillow's image module -- the namespace whose ``fromarray`` builds a pixel array into
# an image -- and it is the boundary's surface for that one conversion, which a caller that
# computed its own pixels (``qrcheck.checks.performance``'s decode probe) needs to make. A
# constructor function for those pixels would add a second API, which the Scope paragraph above
# rules out; the declared name is the surface.
from PIL import Image

# The three byte signatures this product ever sees: PNG, JPEG and SVG.  A target's bytes are
# sniffed, never trusted, so mime_type() answers application/octet-stream for anything else and
# the adapter layer decides what that means (TargetContractError) — this module raises
# nothing for garbage.
PNG_MAGIC: bytes = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC: bytes = b"\xff\xd8\xff"
# SVG is text, so it is recognised by a leading declaration or root element after leading
# whitespace and an optional UTF-8 byte-order mark are stripped.
SVG_PREFIXES: tuple[bytes, ...] = (b"<?xml", b"<svg")

MIME_PNG: str = "image/png"
MIME_JPEG: str = "image/jpeg"
MIME_SVG: str = "image/svg+xml"
MIME_OCTET_STREAM: str = "application/octet-stream"

# mime type -> every prefix that identifies it.  A caller that sniffs SVG for itself must strip
# leading whitespace and a byte-order mark first; _looks_like_svg does exactly that, and
# mime_type() falls through to it for those forms.
MAGIC: dict[str, tuple[bytes, ...]] = {
    MIME_PNG: (PNG_MAGIC,),
    MIME_JPEG: (JPEG_MAGIC,),
    MIME_SVG: SVG_PREFIXES,
}

_UTF8_BOM: bytes = b"\xef\xbb\xbf"
_WHITESPACE: bytes = b" \t\r\n\v\f"

# Where a PNG states its own canvas: the eight-byte signature, then the four-byte length and the
# four-byte type of the IHDR chunk, then that chunk's first two fields, the width and the height
# as big-endian uint32 (ISO/IEC 15948 §5.6 fixes the chunk layout and §11.2.2 the IHDR field
# order).  A conforming stream carries IHDR first, so a 24-byte prefix holds everything
# :func:`probe_image_size` needs to answer, without the image library being consulted at all.
_PNG_IHDR_TYPE_AT: int = 12
_PNG_IHDR_WIDTH_AT: int = 16
_PNG_IHDR_HEIGHT_AT: int = 20
_PNG_IHDR_PREFIX_BYTES: int = 24


class DeclaredCanvasTooLarge(ValueError):
    """An artefact whose declared canvas the image library refuses to open as a decompression bomb.

    A named member of the :exc:`ValueError` family, so a caller can tell "the target declares a
    canvas no ceiling this framework holds would ever accept" from the programming errors
    :func:`probe_image_size` raises for its own arguments. Pillow raises
    :exc:`PIL.Image.DecompressionBombError` from inside ``Image.open`` for a header declaring more
    than twice ``Image.MAX_IMAGE_PIXELS``, before any pixel is allocated, and
    :func:`probe_image_size` converts it into this so a target's own bytes can never surface as a
    library fault.
    """


def _png_declared_size(payload: bytes) -> tuple[int, int] | None:
    """Return the canvas `payload`'s own IHDR chunk declares, or ``None`` when it carries none.

    The dimensions are read straight out of the bytes at :data:`_PNG_IHDR_WIDTH_AT` and
    :data:`_PNG_IHDR_HEIGHT_AT`, each a big-endian ``uint32``, so the answer is the target's own
    declaration and no library guard can pre-empt it. ``None`` means the prefix is not a PNG header
    in the layout this function reads, which leaves the caller to the image library.
    """
    if len(payload) < _PNG_IHDR_PREFIX_BYTES:
        return None
    if payload[_PNG_IHDR_TYPE_AT : _PNG_IHDR_TYPE_AT + 4] != b"IHDR":
        return None
    width = int.from_bytes(payload[_PNG_IHDR_WIDTH_AT : _PNG_IHDR_WIDTH_AT + 4], "big")
    height = int.from_bytes(payload[_PNG_IHDR_HEIGHT_AT : _PNG_IHDR_HEIGHT_AT + 4], "big")
    return width, height


@dataclass(frozen=True)
class OpenImage:
    """One image that is open right now, as reported by :func:`open_images`.

    Frozen, so records compare by value: two opens of the same path for the same case compare equal,
    which is what lets ``CL-ARTIFACTS`` report a leak as a plain equality against an expected
    snapshot. ``path`` is the path exactly as the caller gave it, never resolved, because an
    absolute path must not reach a report or an artefact; ``case_id`` is the case that opened the
    image, or ``None`` when not attributable.
    """

    path: str
    case_id: str | None = None


# Images that are open right now, each keyed by an opaque token created per open, so removing an
# entry removes exactly the one its context created even when two identical paths are open at
# once.  Discovery order never leaks: open_images() sorts.
_OPEN_IMAGES: dict[object, OpenImage] = {}


def _looks_like_svg(data: bytes) -> bool:
    """Return whether ``data`` starts with an SVG declaration or root element.

    Strips an optional UTF-8 byte-order mark and then any leading whitespace before testing the
    prefixes in :data:`SVG_PREFIXES`, because a vendor-written SVG file may begin with either.
    """
    if data.startswith(_UTF8_BOM):
        data = data[len(_UTF8_BOM) :]
    return data.lstrip(_WHITESPACE).startswith(SVG_PREFIXES)


def mime_type(data: bytes) -> str:
    """Return the MIME type implied by the leading bytes of ``data``.

    Answers :data:`MIME_PNG`, :data:`MIME_JPEG`, :data:`MIME_SVG`, or :data:`MIME_OCTET_STREAM` when
    nothing matches. Sniffing never raises on arbitrary bytes: a target may return garbage, and
    calling that a contract failure is the adapter's decision, not this module's. Only a
    non-bytes-like ``data`` raises :exc:`ValueError`, a programming error rather than a finding.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"mime_type expects bytes, got {type(data).__name__}")
    head = bytes(data)
    # A signature beginning at byte 0 is answered by the table; the SVG forms that carry a
    # byte-order mark or leading whitespace need the strip _looks_like_svg performs, so they
    # fall through to it.
    for mime, prefixes in MAGIC.items():
        if head.startswith(prefixes):
            return mime
    if _looks_like_svg(head):
        return MIME_SVG
    return MIME_OCTET_STREAM


def encode_png(image: Image.Image) -> bytes:
    """Encode ``image`` as PNG bytes and return them.

    The encoding is reproducible: no ancillary chunk is written (no ``tIME``, no ``dpi``/``text``
    metadata), so two calls on equal images return identical bytes and, inside the pinned
    Pillow/zlib build, so does a call in another process -- which is what makes a damage artefact
    regenerable from its seed. Nothing is written to disk, and a non-image ``image`` raises
    :exc:`ValueError`.
    """
    if not isinstance(image, Image.Image):
        raise ValueError(f"encode_png expects a PIL.Image.Image, got {type(image).__name__}")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def decode_image(data: bytes) -> Image.Image:
    """Open image bytes and return a fully loaded :class:`PIL.Image.Image`.

    This is how a target's ``GeneratedSymbol.image_bytes`` become an image. The pixels are read
    eagerly, so the returned image keeps no file handle alive and the open-handle snapshot
    ``CL-ARTIFACTS`` takes stays meaningful. Pillow's own exceptions propagate unchanged -- the
    adapter maps them to ``TargetContractError``, and this module deliberately does not know that
    type exists. Raises :exc:`ValueError` when ``data`` is empty or not bytes-like,
    :exc:`PIL.UnidentifiedImageError` when Pillow cannot recognise the content, and :exc:`OSError`
    for truncated or otherwise broken image data.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"decode_image expects bytes, got {type(data).__name__}")
    payload = bytes(data)
    if not payload:
        raise ValueError("decode_image requires non-empty image bytes")
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


def probe_image_size(data: bytes) -> tuple[int, int]:
    """Return the ``(width, height)`` the image bytes declare, without loading any pixels.

    The header-only counterpart of :func:`decode_image`, for a caller that must decide whether the
    pixels are worth allocating at all. A PNG's dimensions are taken from its own IHDR chunk by
    :func:`_png_declared_size`, never from the image library: a header declaring a canvas past
    ``PIL.Image.MAX_IMAGE_PIXELS`` would otherwise make ``Image.open`` raise its decompression-bomb
    guard while the header is being read, so a fact about the target's own bytes would surface as a
    library fault and pre-empt whatever ceiling the caller holds. Every other format is probed
    through Pillow, which reads the dimensions out of its header and closes the image in the
    ``with`` block, so no pixel buffer is created either way. ``data`` is a target's bytes and may
    be hostile or broken, so Pillow's own exceptions propagate unchanged for the caller to
    translate: :exc:`ValueError` when ``data`` is empty or not bytes-like,
    :exc:`DeclaredCanvasTooLarge` (a :exc:`ValueError`) when the declared canvas is one Pillow
    refuses to open as a decompression bomb, :exc:`PIL.UnidentifiedImageError` when Pillow cannot
    recognise the content, and :exc:`OSError` for a truncated or otherwise broken header.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"probe_image_size expects bytes, got {type(data).__name__}")
    payload = bytes(data)
    if not payload:
        raise ValueError("probe_image_size requires non-empty image bytes")
    if payload.startswith(PNG_MAGIC):
        declared = _png_declared_size(payload)
        if declared is not None:
            return declared
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            width, height = probe.size
    except Image.DecompressionBombError as exc:
        raise DeclaredCanvasTooLarge(
            f"the {mime_type(payload)} artefact declares a canvas beyond the "
            f"{Image.MAX_IMAGE_PIXELS} pixel limit PIL.Image.MAX_IMAGE_PIXELS sets, which is what "
            f"Pillow refuses to open: {exc}"
        ) from exc
    return int(width), int(height)


def composite_overlay(symbol: Image.Image, overlay: Image.Image) -> Image.Image:
    """Return a new RGBA image with ``overlay`` composited centred on ``symbol``.

    The one transform this module performs, and it exists because of a capability contract:
    ``logo_overlay`` is a member the built-in target declares, so a request that carries an overlay
    must reach a symbol carrying it -- a declared capability with no behaviour behind it is
    indistinguishable from an ignored one. The overlay keeps its own pixel size, is centred on both
    axes and is clipped when it is larger than the symbol in a dimension; its alpha channel is
    honoured, so a semi-transparent overlay blends with the modules under it. Neither argument is
    mutated and no file is read or written: the symbol is converted into a fresh image that becomes
    the canvas, and the result carries no text chunk or timestamp, so equal inputs composite to
    equal pixels. Raises :exc:`ValueError` when either argument is not a :class:`PIL.Image.Image`.
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
    # ``convert`` returns a new image even when the mode already matches, so the canvas below
    # can be pasted onto without touching either argument.
    base = symbol.convert("RGBA")
    patch = overlay.convert("RGBA")
    offset = ((base.width - patch.width) // 2, (base.height - patch.height) // 2)
    # A negative offset is how Pillow clips an overlay that overhangs the symbol, so the
    # centred placement needs no crop of its own.
    base.paste(patch, offset, patch)
    return base


def encode_base64(data: bytes) -> str:
    """Return ``data`` base64-encoded as ASCII text, unwrapped.

    No newlines and no line wrapping: the data URI built by :func:`to_data_uri` and the corpus'
    ``payload_b64`` field both require the single-line form.
    """
    return base64.b64encode(bytes(data)).decode("ascii")


def decode_base64(text: str) -> bytes:
    """Decode base64 ``text`` to bytes, rejecting malformed input.

    Decoding is strict, so a malformed body raises :exc:`binascii.Error` instead of quietly decoding
    a prefix of it; the adapter converts that into ``TargetContractError`` and this module lets it
    propagate unchanged. ``b""`` decodes to ``b""``, and non-ASCII text raises :exc:`ValueError`.
    """
    return base64.b64decode(text, validate=True)


def to_data_uri(data: bytes) -> str:
    """Return ``data`` as an inline ``data:`` URI.

    The only way an image reaches ``index.html``: the report embeds images inline, so it holds no
    ``link`` or ``img`` pointing at an external origin and opens without network access. The result
    is a pure function of ``data`` -- ``data:<mime>;base64,<payload>``, with ``<mime>`` from
    :func:`mime_type` and no whitespace, quoting or origin beyond the ``data:`` scheme.
    """
    return f"data:{mime_type(data)};base64,{encode_base64(data)}"


@contextmanager
def open_image(path: str | Path, *, case_id: str | None = None) -> Iterator[Image.Image]:
    """Open the image at ``path`` and register it as open for the life of the context.

    The image is registered before it is yielded, so :func:`open_images` reports it while the
    ``with`` body runs and no longer afterwards, whether the body returned or raised:
    ``CL-ARTIFACTS`` reads that snapshot to attribute an unclosed handle to the case that leaked it,
    so every image the framework touches is opened through here. ``path`` is kept exactly as given,
    the registration is dropped in a ``finally`` block together with closing the image, so a handle
    cannot outlive the context, and ``case_id`` names the case that opened it or ``None`` when the
    open is not attributable. Raises :exc:`PIL.UnidentifiedImageError` when Pillow cannot recognise
    the content, and :exc:`OSError` when the file cannot be opened or read.
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
    end of a run.  Entries are ordered by ``(case_id or "", path)``, so a report of a
    leak is identical on every run.

    Returns:
        The currently open images, ordered by case identifier and then by path.
    """
    return tuple(sorted(_OPEN_IMAGES.values(), key=lambda entry: (entry.case_id or "", entry.path)))


__all__ = [
    "DeclaredCanvasTooLarge",
    "Image",
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
    "probe_image_size",
    "to_data_uri",
]
