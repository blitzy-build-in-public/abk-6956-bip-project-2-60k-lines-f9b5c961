"""The ``library`` adapter: an in-process import-and-call target.

A ``type = "library"`` target is a QR generator the framework calls as a plain
Python function in its own interpreter.  Everything this module knows about that
generator arrives through the ``[targets.<name>]`` table -- the import path, the
mapping from framework fields to the callable's keyword arguments, how the
returned value carries the artefact, and how the callable's own metadata is read
-- so the adapter names no generator library and imports none.  That is the
strongest form of the rule that no vendor library is hard-coded anywhere in the
package: the adapter drives whatever callable the configuration names, so
pointing the framework at a different generator is a configuration table rather
than a code change.

The four configuration keys
----------------------------
``callable``
    ``"module:attribute"``.  The attribute may be a dotted path
    (``"module:Factory.build"``), in which case every segment is walked.  The
    module is resolved **lazily**, on the first :meth:`LibraryAdapter.generate`,
    so constructing an adapter costs nothing and ``qrcheck doctor``'s single
    probe generate -- not the constructor -- is what proves the path exists.  A
    missing module or attribute raises :class:`TargetUnavailable`; a raw
    ``ImportError`` never escapes, because the runner reports one ``error``
    result per affected check on target loss and a raw import failure would
    instead surface as a framework crash that misattributes the fault.
``kwargs_map``
    Framework field name -> the callable's keyword-argument name.  The framework
    field vocabulary is exactly the field list of :class:`GenerateRequest`
    (``payload``, ``ec_level``, ``output_format``, ``module_size_px``,
    ``quiet_zone_modules``, ``foreground_rgb``, ``background_rgb``,
    ``logo_overlay_path``, ``version``, ``mask``), read here from the dataclass
    itself rather than restated.  **Only mapped fields are passed**, so a target
    receives exactly the arguments its configuration declares and an unmapped
    field is never inferred: a generator that takes no version pin simply maps
    none.
``returns``
    How the returned value carries the artefact, decoded by the one shared
    implementation in :func:`qrcheck.targets.base.decode_returned_image`:
    ``raw_bytes`` (the bytes themselves), ``base64`` (base64 text),
    ``bytes_path`` (a path the target chose) or ``file_path`` (a path the target
    wrote).  A ``library`` target has no framework-supplied output path in the
    request vocabulary, so both path kinds read the path the callable returns.
``metadata_map``
    Optional.  Framework metadata field (``version``, ``ec``, ``mask``, ``mode``)
    -> the key or dotted path carrying it in the metadata the callable returns.
    Declaring it changes the callable's return shape, as described next.

Return shape
------------
Without ``metadata_map`` the callable returns the image value itself, and the
adapter derives the artefact bytes from it through ``returns``.

With ``metadata_map`` the callable returns the 2-tuple
``(image_value, metadata_mapping)``, where ``metadata_mapping`` is any mapping --
typically a dict keyed by the names the configuration gives.  This is the only
channel a ``library`` target has for declaring which version, error-correction
level, mask and mode it used, and every value it carries is **evidence only,
never the assertion**: ``SY-EC-MATCH`` reads the level from the decoder and from
the symbol's format information, ``SY-MASK`` from the format information and
``SY-VERSION-MIN`` from the sampled module count, so a target that misreports
its metadata is recorded as having misreported it rather than believed.  A
return value of any other shape raises :class:`TargetContractError`.

The rejection channel
---------------------
A target the framework does not own declares a client-side refusal by raising an
exception whose text begins with ``rejected:``; the remainder, with the
separating whitespace removed, becomes the message of
:class:`TargetRejected` verbatim.  That is what lets a configured target satisfy
``IV-EMPTY`` and ``IV-BAD-OPTION`` -- which require a declared rejection of a
request the framework considers valid -- instead of appearing to crash.  Any
other exception from the callable becomes :class:`TargetContractError` carrying
``"<ExceptionType>: <text>"``, so a target bug is reported as a target bug.

``timeout_s`` is accepted and cannot be enforced
------------------------------------------------
The configured ``targets.<name>.timeout_s`` is stored on the adapter and enforced
nowhere, because a call inside this interpreter has no process boundary, socket
or pipe to interrupt.  Enforcing it would need a thread, a signal alarm or a
process pool, and the product admits no such component: execution is
single-process and sequential so that a run stays deterministic.  A library
target that hangs is a target defect the run reports as a slow case, and
``docs/targets.md`` states the limitation where a user configures a target.  The
``cli`` adapter enforces the same key with ``subprocess.run(timeout=...)`` and
the ``http`` adapter with ``httpx.Timeout``.

What this module imports
------------------------
The standard library, :mod:`qrcheck.config.models` for the record it is handed,
and the shared helpers of :mod:`qrcheck.targets.base`.  No image library, no
decoder and no generator library, and nothing from ``qrcheck.checks``,
``qrcheck.report``, ``qrcheck.decode`` or ``qrcheck.registry``: the adapter sits
at the port and knows only the port's vocabulary.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping
from dataclasses import fields
from types import MappingProxyType
from typing import Final, cast

from qrcheck.config.models import LibraryTarget, ReturnsKind
from qrcheck.targets.base import (
    Capability,
    GenerateRequest,
    GeneratedSymbol,
    TargetContractError,
    TargetError,
    TargetRejected,
    TargetUnavailable,
    apply_metadata_map,
    declared_capabilities,
    decode_returned_image,
    ensure_image_bytes,
    require_capabilities,
)

__all__ = ["LibraryAdapter"]

# A client-side refusal is recognised by this prefix, matched case-sensitively at
# the start of the exception's text, exactly as ``docs/targets.md`` documents it.
_REJECTION_PREFIX: Final[str] = "rejected:"

# The framework field vocabulary is the field list of :class:`GenerateRequest`
# itself, so a field added to the request record becomes mappable here with no
# second edit and the two can never drift apart.
_REQUEST_FIELDS: Final[tuple[str, ...]] = tuple(
    field.name for field in fields(GenerateRequest)
)

# The capabilities a request is *inferable* to need, keyed by the request field
# that asks for them.  ``png``/``svg`` come from ``output_format``; the rest are
# the optional pins and overlays, which are ``None`` in an ordinary request and
# therefore say positively that the operation was asked for.  ``ec_level`` and
# the colours are deliberately absent: both are always present in a
# :class:`GenerateRequest`, with defaults, so no request can ask for them here --
# a check that needs them declares that in its own ``requires`` set, which is
# what ``qrcheck/registry/resolve.py`` intersects with the declared capabilities.
# Deriving them from a request would demand a capability the caller never asked
# for and would skip a check the target can actually run.

class LibraryAdapter:
    """Drive one configured in-process QR generator through the adapter port.

    The port's three runtime members are all here: ``name`` and ``capabilities``
    are what the registry and the runner read, and :meth:`generate` is what a
    check calls.  The capability set is exactly the one the configuration
    declares -- never a built-in superset -- because a shortfall against a
    check's requirement has to become ``skipped``/``unsupported_capability``,
    counted separately and never a pass, rather than a silent wrong assertion.

    ``timeout_s`` is recorded and not enforced; the module docstring says why.
    """

    def __init__(self, config: LibraryTarget, *, name: str | None = None) -> None:
        """Record the resolved configuration; resolve no import path yet.

        Args:
            config: The resolved ``[targets.<name>]`` record of a
                ``type = "library"`` target.
            name: The adapter name every diagnostic quotes.  Defaults to
                ``config.type``, so a factory call and a name-passing call read
                the same way.

        Raises:
            ValueError: ``kwargs_map`` names a framework field that is not a
                :class:`GenerateRequest` field, or maps two framework fields onto
                one callable kwarg.  ``qrcheck.config.schema`` validates that
                every key and value of the mapping is text at load time but not
                its vocabulary, so a configuration reaching here has been past
                the schema gate and the divergence is a framework defect: one
                named failure beats one silent wrong call.
        """
        self.name: str = name if name is not None else config.type
        self.capabilities: frozenset[Capability] = declared_capabilities(config)
        self.callable: str = config.callable
        self.kwargs_map: Mapping[str, str] = MappingProxyType(dict(config.kwargs_map))
        self.returns: ReturnsKind = config.returns
        self.metadata_map: Mapping[str, str] = MappingProxyType(dict(config.metadata_map))
        self.timeout_s: float = config.timeout_s
        self._target_name: str = config.name
        self._mapped: tuple[tuple[str, str], ...] = _checked_kwargs_map(
            self.kwargs_map, self._key
        )
        self._resolved: Callable[..., object] | None = None

    def __repr__(self) -> str:
        """Return a diagnostic form naming the adapter and its import path."""
        return f"{type(self).__name__}(name={self.name!r}, callable={self.callable!r})"

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Call the configured generator once and return its symbol.

        The call is made with exactly the keyword arguments ``kwargs_map``
        declares, in :class:`GenerateRequest` field order, and its duration is
        measured around the call alone -- resolution, decoding and metadata
        projection are framework work and are excluded from ``elapsed_ms``.

        Args:
            request: The request to answer.

        Returns:
            The artefact bytes exactly as the callable produced them, the format
            the request asked for, the metadata the callable declared (all four
            fields ``None`` when no ``metadata_map`` is configured) and the
            call's duration in milliseconds.

        Raises:
            TargetUnsupported: The request asks for a capability this adapter did
                not declare.  Raised by :func:`require_capabilities` before any
                import, call or file read, so an unsupported request never
                reaches the target.
            TargetUnavailable: The configured ``targets.<name>.callable`` cannot
                be imported or cannot be found in its module.
            TargetRejected: The callable declared a client-side rejection through
                the ``rejected:`` channel; the message is the target's text.
            TargetContractError: The callable raised anything else, returned a
                value shaped wrongly for ``returns`` or ``metadata_map``, named a
                metadata field the returned mapping does not carry, or produced
                bytes that are not the artefact the request asked for.
            ValueError: The configuration the adapter was built from is malformed
                in a way the schema should have rejected -- an import path that is
                not ``module:attribute``, a ``returns`` kind outside its enum, or a
                ``metadata_map`` naming a field outside the four metadata fields.
                A framework defect rather than a target one, so it is deliberately
                not reported as a target failure.

        ``TargetTimeout`` is never raised: the target runs in this interpreter, so
        there is no boundary to interrupt at ``timeout_s``.
        """
        # Belt-and-braces on the registry's resolution: a check whose requirements
        # the declared set cannot satisfy is already skipped before execution, so
        # this gate only ever fires for a request that reaches the adapter out of
        # order -- and it fires before the import, the call or any file read.
        require_capabilities(self, _requested_capabilities(request))
        func = self._resolve()
        kwargs = {
            target_name: getattr(request, field_name)
            for field_name, target_name in self._mapped
        }
        started = time.perf_counter()
        try:
            returned: object = func(**kwargs)
        except TargetError:
            # Already inside the adapter vocabulary: a target that speaks it gets
            # its message through untouched rather than re-wrapped as a defect.
            raise
        except Exception as exc:
            raise _convert_call_failure(exc, self._key) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        image_value, raw_metadata = self._split_return(returned)
        data = ensure_image_bytes(
            decode_returned_image(image_value, self.returns, base_dir=None),
            request.output_format,
        )
        declared = apply_metadata_map(
            {} if raw_metadata is None else raw_metadata, self.metadata_map
        )
        return GeneratedSymbol(
            image_bytes=data,
            output_format=request.output_format,
            declared_version=cast("int | None", declared.get("version")),
            declared_ec=cast("str | None", declared.get("ec")),
            declared_mask=cast("int | None", declared.get("mask")),
            declared_mode=cast("str | None", declared.get("mode")),
            elapsed_ms=elapsed_ms,
        )

    def close(self) -> None:
        """Drop the cached callable; safe to call twice and after a failure.

        The adapter holds no socket, file or child process, so closing releases
        the resolved callable and nothing else.  A later :meth:`generate`
        re-resolves it, which is what makes "close, then use again" behave like a
        fresh adapter instead of raising: the runner closes every adapter once a
        run ends, including one it closed after an error.
        """
        self._resolved = None

    @property
    def _key(self) -> str:
        """Return the dotted configuration key every diagnostic about it names."""
        return f"targets.{self._target_name}.callable"

    def _resolve(self) -> Callable[..., object]:
        """Return the configured callable, importing it on first use.

        Raises:
            TargetUnavailable: The module cannot be imported, or an attribute
                segment is absent from it.
            TargetContractError: The attribute exists but is not callable.
            ValueError: ``callable`` is not a ``module:attribute`` path, which the
                configuration schema should have refused at load time.
        """
        resolved = self._resolved
        if resolved is not None:
            return resolved

        module_path, attribute_path = _split_callable(self.callable, self._key)
        try:
            target: object = importlib.import_module(module_path)
        except Exception as exc:
            # ImportError for a missing module, and anything the module's own body
            # raises: either way the configured target cannot be loaded, which is
            # the TargetUnavailable trigger.  The cause is quoted, never swallowed.
            raise TargetUnavailable(
                f"{self._key} = {self.callable!r}: cannot import module "
                f"{module_path!r} to resolve attribute {attribute_path!r} "
                f"({type(exc).__name__}: {exc})"
            ) from exc

        walked: list[str] = []
        for segment in attribute_path.split("."):
            walked.append(segment)
            try:
                target = getattr(target, segment)
            except AttributeError as exc:
                raise TargetUnavailable(
                    f"{self._key} = {self.callable!r}: module {module_path!r} has no "
                    f"attribute {'.'.join(walked)!r} ({exc})"
                ) from exc

        if not callable(target):
            raise TargetContractError(
                f"{self._key} = {self.callable!r}: resolves to "
                f"{type(target).__name__}, which is not callable"
            )
        func = cast("Callable[..., object]", target)
        self._resolved = func
        return func

    def _split_return(self, returned: object) -> tuple[object, Mapping[str, object] | None]:
        """Split the callable's return value into the image value and its metadata.

        Args:
            returned: The value the callable returned.

        Returns:
            The image value, and the metadata mapping or ``None`` when no
            ``metadata_map`` is configured.

        Raises:
            TargetContractError: ``metadata_map`` is configured and the return
                value is not a 2-element sequence whose second element is a
                mapping.
        """
        if not self.metadata_map:
            return returned, None
        if not isinstance(returned, (tuple, list)) or len(returned) != 2:
            raise TargetContractError(
                f"{self._key}: metadata_map is configured, so the callable must return "
                f"(image_value, metadata); it returned {type(returned).__name__}"
            )
        image_value, metadata = returned
        if not isinstance(metadata, Mapping):
            raise TargetContractError(
                f"{self._key}: metadata_map is configured, so the second element the "
                f"callable returns must be a mapping; it returned "
                f"{type(metadata).__name__}"
            )
        return image_value, cast("Mapping[str, object]", metadata)


def _requested_capabilities(request: GenerateRequest) -> frozenset[Capability]:
    """Return the capabilities `request` itself asks for.

    Only a field that positively asks for an operation can be read off a request:
    the output format, and the version pin, mask pin, module size, quiet zone and
    logo overlay, each of which is ``None`` unless the caller asked for it.  See
    the module-level comment on the inference for why ``ec_level`` and the colours
    are not among them.

    Returns:
        The capabilities to require before the call, empty for an ordinary
        default request.
    """
    needed: set[Capability] = {Capability(request.output_format)}
    if request.module_size_px is not None:
        needed.add(Capability.module_size)
    if request.quiet_zone_modules is not None:
        needed.add(Capability.quiet_zone)
    if request.logo_overlay_path is not None:
        needed.add(Capability.logo_overlay)
    if request.version is not None:
        needed.add(Capability.version_pin)
    if request.mask is not None:
        needed.add(Capability.mask_pin)
    return frozenset(needed)


def _convert_call_failure(exc: Exception, key: str) -> TargetError:
    """Return the adapter error that stands for an exception the callable raised.

    Args:
        exc: The exception the callable raised.
        key: The configured key the diagnostic names.

    Returns:
        :class:`TargetRejected` carrying the text after a ``rejected:`` prefix, or
        :class:`TargetContractError` carrying ``"<ExceptionType>: <text>"``.

    Only the whitespace separating the prefix from the message is removed: the
    rest is the target's text verbatim, because ``IV-*`` checks record it in
    ``decoded.rejection`` and every other check reports it as reason
    ``target_rejected``.
    """
    text = str(exc)
    if text.startswith(_REJECTION_PREFIX):
        return TargetRejected(text[len(_REJECTION_PREFIX) :].strip())
    return TargetContractError(f"{type(exc).__name__}: {text}")


def _checked_kwargs_map(
    kwargs_map: Mapping[str, str], key: str
) -> tuple[tuple[str, str], ...]:
    """Return the kwarg mapping as pairs in :class:`GenerateRequest` field order.

    The order is the request record's own field order rather than the TOML table's
    insertion order, so the call a target receives is identical whatever order the
    configuration lists its keys in.

    Args:
        kwargs_map: Framework field name -> the callable's keyword argument name.
        key: The configured key a violation names.

    Returns:
        One ``(framework field, callable kwarg)`` pair per mapped field.

    Raises:
        ValueError: A framework field is outside the request vocabulary, or two
            framework fields map onto one callable kwarg and would overwrite each
            other silently.
    """
    unknown = sorted(set(kwargs_map) - set(_REQUEST_FIELDS))
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ValueError(
            f"{key}: kwargs_map names unknown framework field(s) {names}; expected any "
            f"of {', '.join(_REQUEST_FIELDS)}"
        )
    claimed: dict[str, str] = {}
    for field_name in _REQUEST_FIELDS:
        if field_name not in kwargs_map:
            continue
        target_name = kwargs_map[field_name]
        if target_name in claimed:
            raise ValueError(
                f"{key}: kwargs_map maps both {claimed[target_name]!r} and "
                f"{field_name!r} to the callable kwarg {target_name!r}; one argument "
                "cannot carry two framework fields"
            )
        claimed[target_name] = field_name
    return tuple(
        (field_name, kwargs_map[field_name])
        for field_name in _REQUEST_FIELDS
        if field_name in kwargs_map
    )


def _split_callable(text: str, key: str) -> tuple[str, str]:
    """Split ``"module:attribute"`` into its module path and attribute path.

    Args:
        text: The configured ``callable`` value.
        key: The configured key a violation names.

    Returns:
        The module path and the (possibly dotted) attribute path.

    Raises:
        ValueError: The value is not exactly one ``module:attribute`` of Python
            identifiers.  ``qrcheck.config.schema`` performs the same check when it
            loads the configuration and exits 2 naming the key, so a value reaching
            here has been past that gate and the divergence is a framework defect.
    """
    if text.count(":") != 1:
        raise ValueError(
            f"{key} = {text!r}: expected exactly one ':' as 'module:attribute'"
        )
    module_path, attribute_path = text.split(":")
    if not module_path or not attribute_path:
        raise ValueError(
            f"{key} = {text!r}: expected both parts of 'module:attribute'"
        )
    for part in (*module_path.split("."), *attribute_path.split(".")):
        if not part.isidentifier():
            raise ValueError(
                f"{key} = {text!r}: {part!r} is not a valid Python identifier"
            )
    return module_path, attribute_path
