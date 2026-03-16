# Principles

## Primary Objective

Build this repository into a rigorous AI quant research stack.

Do not optimize for demo value or superficial automation if that delays real model validation.

## Priority Order

1. Research rigor
2. Honest evaluation
3. AI alpha model quality
4. Paper trading automation
5. Live trading concerns

## Required Research Standards

- prevent forward leakage
- prefer strict out-of-sample evaluation
- keep training and evaluation windows clearly separated
- compare every new model against a strong non-AI baseline
- distinguish feature engineering improvements from model-family improvements
- report both returns and risk metrics

## What Counts As "AI" Here

Acceptable AI model classes for this project include:

- MLP and other feed-forward supervised models
- LSTM or similar sequence models
- transformer-based time-series models
- LLM-driven text signals
- reinforcement learning for sizing or allocation after strong baselines exist

Heuristic keyword scoring does not count as the project's core AI model.

## Reporting Rules

Every serious experiment should say:

- what the model is
- what the label is
- what the training window is
- what the out-of-sample window is
- what the baseline is
- whether the result is backtest, paper, or live

## Anti-Goals

- do not confuse platform completeness with alpha quality
- do not chase RL early
- do not rely on LLM prompts alone as the core quant model
- do not claim paper-grade or professional rigor prematurely
