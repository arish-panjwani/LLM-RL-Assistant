# Prompt Optimization

This folder handles transforming raw user input (text or image-derived) into optimized prompts.

## Modules
- `rewriter.py` – Rewrites prompts using RL agent output
- `lexical_metrics.py` – Analyzes clarity, redundancy, cosine similarity
- `fact_checker.py` – Uses Google, Wikipedia, or LLM self-checking for factual validation

## Image Handling
- Accepts text extracted from uploaded images (OCR or captioning)
- Passes this to rewriter and fact-checker modules like regular text

## Purpose
- Reduce ambiguity and improve clarity of LLM input prompts
- Penalize hallucination-prone or vague input patterns
