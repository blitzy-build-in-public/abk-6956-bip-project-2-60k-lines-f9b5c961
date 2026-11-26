"""Deterministic QR payload corpus: capacity tables, payload classes, the 240-record builder,
and the canonical JSONL format.

Importing this package performs no work: it re-exports the public API of
``qrcheck.corpus.builder`` and ``qrcheck.corpus.io``.
"""

from qrcheck.corpus.builder import CorpusRecord, build_corpus
from qrcheck.corpus.io import filter_by_class, load_corpus, write_corpus

# The package's public surface, written out sorted so ``import *``, ``dir()`` and any surface
# assertion are stable across runs and machines (AAP §0.5.3).
__all__ = ("CorpusRecord", "build_corpus", "filter_by_class", "load_corpus", "write_corpus")
