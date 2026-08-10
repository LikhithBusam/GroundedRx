"""rrf_fuse is pure (no Qdrant/BM25/network) -- these tests use hand-built
dicts, no live vector store or GPU needed."""

from groundedrx.retrieval import rrf_fuse


def test_dense_only_hit_gets_zero_bm25_score():
    dense = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.7}]
    result = rrf_fuse(dense, sparse_hits=[], rrf_k=60)
    assert [r["id"] for r in result] == ["a", "b"]
    assert result[0]["bm25_score"] == 0.0
    assert result[0]["score"] == 0.9


def test_sparse_only_hit_gets_zero_dense_score():
    sparse = [{"id": "x", "bm25_score": 5.0}]
    result = rrf_fuse(dense_hits=[], sparse_hits=sparse, rrf_k=60)
    assert result[0]["id"] == "x"
    assert result[0]["score"] == 0.0
    assert result[0]["bm25_score"] == 5.0


def test_chunk_found_by_both_ranks_above_either_alone():
    # "b" appears in both lists (rank 1 in each) and should outrank a
    # same-single-list top rank once both RRF contributions are summed.
    dense = [{"id": "a", "score": 0.95}, {"id": "b", "score": 0.80}]
    sparse = [{"id": "b", "bm25_score": 9.0}, {"id": "c", "bm25_score": 8.0}]
    result = rrf_fuse(dense, sparse, rrf_k=60)
    assert result[0]["id"] == "b"
    assert result[0]["score"] == 0.80
    assert result[0]["bm25_score"] == 9.0


def test_fusion_is_rank_based_not_score_based():
    # BM25's raw score (1000.0) dwarfs cosine (0-1 bounded), but fusion must
    # not let that unbounded scale dominate -- rank 0 in either list
    # contributes the same 1/(rrf_k+1), regardless of the raw score gap.
    dense = [{"id": "a", "score": 0.99}]
    sparse = [{"id": "z", "bm25_score": 1000.0}]
    result = rrf_fuse(dense, sparse, rrf_k=60)
    # both are rank-0 in their own list -> identical RRF contribution -> tie,
    # dense inserted first so it wins any stable-sort tie
    assert {r["id"] for r in result} == {"a", "z"}
    assert result[0]["id"] == "a"


def test_empty_inputs_return_empty():
    assert rrf_fuse([], [], rrf_k=60) == []
