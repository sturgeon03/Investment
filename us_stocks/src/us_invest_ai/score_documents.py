from __future__ import annotations

import argparse

from us_invest_ai.llm_scoring import (
    HeuristicScorer,
    OpenAICompatibleConfig,
    OpenAICompatibleScorer,
    aggregate_document_scores,
    load_documents,
    save_score_outputs,
    score_documents,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score text documents into daily LLM equity signals.")
    parser.add_argument(
        "--input-csv",
        default="us_stocks/documents/sec_sections.csv",
        help="CSV file with columns date,ticker,title,text and optional doc_type,source.",
    )
    parser.add_argument(
        "--detailed-output",
        default="us_stocks/signals/llm_document_scores.csv",
        help="Detailed per-document output CSV path.",
    )
    parser.add_argument(
        "--aggregated-output",
        default="us_stocks/signals/llm_scores.generated.csv",
        help="Aggregated date,ticker,llm_score CSV path used by the backtest.",
    )
    parser.add_argument(
        "--provider",
        default="openai-compatible",
        choices=["heuristic", "openai-compatible"],
        help="Use heuristic for local dry runs or openai-compatible for real LLM scoring.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="Base URL for an OpenAI-compatible chat completion API.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="Model name for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout for each document scoring request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the chat completion request.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    documents = load_documents(args.input_csv)

    if args.provider == "heuristic":
        scorer = HeuristicScorer()
    else:
        scorer = OpenAICompatibleScorer(
            OpenAICompatibleConfig(
                base_url=args.base_url,
                model=args.model,
                api_key_env=args.api_key_env,
                timeout_seconds=args.timeout_seconds,
                temperature=args.temperature,
            )
        )

    scored_documents = score_documents(documents, scorer)
    aggregated_scores = aggregate_document_scores(scored_documents)
    save_score_outputs(
        scored_documents=scored_documents,
        aggregated_scores=aggregated_scores,
        detailed_output_path=args.detailed_output,
        aggregated_output_path=args.aggregated_output,
    )

    print(f"Scored {len(scored_documents)} documents.")
    print(f"Aggregated into {len(aggregated_scores)} date/ticker/horizon signals.")
    print(f"Detailed output: {args.detailed_output}")
    print(f"Aggregated output: {args.aggregated_output}")


if __name__ == "__main__":
    main()
