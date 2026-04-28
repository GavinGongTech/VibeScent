from __future__ import annotations

import numpy as np

try:
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


class BM25CorpusScorer:
    def __init__(self, corpus_texts: list[str]) -> None:
        self._n = len(corpus_texts)
        if HAS_BM25:
            tokenized_corpus = [text.lower().split() for text in corpus_texts]
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = None

    @property
    def available(self) -> bool:
        return self._bm25 is not None

    def score(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(self._n, dtype=np.float32)
        tokens = query.lower().split()
        return np.array(self._bm25.get_scores(tokens), dtype=np.float32)
