from __future__ import annotations

import pytest

from vibescents.cli import build_parser


def test_parser_embed_csv_required_args() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "embed-csv",
        "--input-csv", "data.csv",
        "--id-column", "fragrance_id",
        "--text-column", "retrieval_text",
        "--output-dir", "out/",
    ])
    assert args.command == "embed-csv"
    assert args.input_csv == "data.csv"
    assert args.id_column == "fragrance_id"
    assert args.input_type == "document"  # default


def test_parser_embed_csv_optional_model() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "embed-csv",
        "--input-csv", "data.csv",
        "--id-column", "id",
        "--text-column", "text",
        "--output-dir", "out/",
        "--model", "voyage-3",
        "--input-type", "query",
    ])
    assert args.model == "voyage-3"
    assert args.input_type == "query"


def test_parser_embed_occasions() -> None:
    parser = build_parser()
    args = parser.parse_args(["embed-occasions", "--input-json", "data.json", "--output-dir", "out/"])
    assert args.command == "embed-occasions"
    assert args.input_json == "data.json"


def test_parser_multimodal_retrieve_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "multimodal-retrieve",
        "--fragrance-csv", "f.csv",
        "--id-column", "id",
        "--text-column", "text",
        "--occasion-text", "Black tie gala",
        "--output-dir", "out/",
    ])
    assert args.command == "multimodal-retrieve"
    assert args.top_k == 10  # default


def test_parser_multimodal_retrieve_custom_top_k() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "multimodal-retrieve",
        "--fragrance-csv", "f.csv",
        "--id-column", "id",
        "--text-column", "text",
        "--occasion-text", "dinner",
        "--output-dir", "out/",
        "--top-k", "5",
        "--image-path", "outfit.jpg",
    ])
    assert args.top_k == 5
    assert args.image_path == "outfit.jpg"


def test_parser_rerank() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "rerank",
        "--candidate-json", "c.json",
        "--occasion-text", "Evening dinner",
        "--output-json", "result.json",
    ])
    assert args.command == "rerank"
    assert args.image_path is None  # optional, default None


def test_parser_generate_benchmark_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "generate-benchmark",
        "--briefs-json", "briefs.json",
        "--output-json", "out.json",
    ])
    assert args.command == "generate-benchmark"
    assert args.runs == 3
    assert args.model is None


def test_parser_missing_required_arg_exits() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["embed-csv"])  # missing --input-csv etc.


def test_parser_no_subcommand_exits() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
