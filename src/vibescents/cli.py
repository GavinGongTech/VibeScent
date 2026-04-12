from __future__ import annotations

import argparse
from pathlib import Path

from vibescents.benchmark import BenchmarkGenerator, consolidate_case_drafts
from vibescents.io_utils import dump_json, ensure_dir, load_dataframe, load_json
from vibescents.pipelines import embed_occasions, embed_text_frame, retrieve_with_multimodal_query
from vibescents.reranker import GeminiReranker
from vibescents.schemas import RetrievalCandidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vibescents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_csv = subparsers.add_parser("embed-csv")
    embed_csv.add_argument("--input-csv", required=True)
    embed_csv.add_argument("--id-column", required=True)
    embed_csv.add_argument("--text-column", required=True)
    embed_csv.add_argument("--output-dir", required=True)
    embed_csv.add_argument("--model")
    embed_csv.add_argument("--input-type", default="document", help="Voyage input_type: 'document' or 'query'")

    embed_occasions_parser = subparsers.add_parser("embed-occasions")
    embed_occasions_parser.add_argument("--input-json", required=True)
    embed_occasions_parser.add_argument("--output-dir", required=True)

    multimodal = subparsers.add_parser("multimodal-retrieve")
    multimodal.add_argument("--fragrance-csv", required=True)
    multimodal.add_argument("--id-column", required=True)
    multimodal.add_argument("--text-column", required=True)
    multimodal.add_argument("--occasion-text", required=True)
    multimodal.add_argument("--image-path")
    multimodal.add_argument("--output-dir", required=True)
    multimodal.add_argument("--top-k", type=int, default=10)

    rerank = subparsers.add_parser("rerank")
    rerank.add_argument("--candidate-json", required=True)
    rerank.add_argument("--occasion-text", required=True)
    rerank.add_argument("--image-path")
    rerank.add_argument("--output-json", required=True)

    benchmark = subparsers.add_parser("generate-benchmark")
    benchmark.add_argument("--briefs-json", required=True)
    benchmark.add_argument("--output-json", required=True)
    benchmark.add_argument("--runs", type=int, default=3)
    benchmark.add_argument("--model", default=None, help="Override the generation model (default: settings.reranker_model)")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "embed-csv":
        frame = load_dataframe(args.input_csv)
        embed_text_frame(
            frame,
            id_column=args.id_column,
            text_column=args.text_column,
            output_dir=args.output_dir,
            model=args.model,
            input_type=args.input_type,
        )
        return

    if args.command == "embed-occasions":
        raw = load_json(args.input_json)
        embed_occasions(raw, output_dir=args.output_dir)
        return

    if args.command == "multimodal-retrieve":
        frame = load_dataframe(args.fragrance_csv)
        retrieve_with_multimodal_query(
            frame,
            id_column=args.id_column,
            text_column=args.text_column,
            occasion_text=args.occasion_text,
            image_path=args.image_path,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )
        return

    if args.command == "rerank":
        raw_candidates = load_json(args.candidate_json)
        candidates = [RetrievalCandidate.model_validate(item) for item in raw_candidates]
        response = GeminiReranker().rerank(
            occasion_text=args.occasion_text,
            image_path=args.image_path,
            candidates=candidates,
        )
        output_path = Path(args.output_json)
        ensure_dir(output_path.parent)
        dump_json(output_path, response.model_dump())
        return

    if args.command == "generate-benchmark":
        briefs = load_json(args.briefs_json)
        settings = None
        if args.model:
            from vibescents.settings import Settings
            settings = Settings(api_key=Settings.from_env().api_key, reranker_model=args.model)
        generator = BenchmarkGenerator(settings=settings)
        consolidated = []
        for item in briefs:
            drafts = generator.generate_case_labels(
                case_id=item["case_id"],
                brief=item["brief"],
                runs=args.runs,
            )
            consolidated.append(consolidate_case_drafts(drafts).model_dump())
        output_path = Path(args.output_json)
        ensure_dir(output_path.parent)
        dump_json(output_path, consolidated)
        return

    raise ValueError(f"Unsupported command: {args.command}")
