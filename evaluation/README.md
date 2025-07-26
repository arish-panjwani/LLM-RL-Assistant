# Evaluation & Metrics

Tracks feedback and model performance using both human and AI-based signals.

## Files
- `metrics_logger.py` – Stores feedback, reward, model performance
- `sentiment_analysis.py` – Uses Vader or LLM to score sentiment
- `dashboard/` – UI code to visualize logs and metrics

## Metrics Tracked
- Cosine similarity of LLM responses
- Lexical redundancy and diversity
- Groq LLM self-evaluation scores
- User ratings and sentiment
- OCR confidence and image tagging accuracy

## Use
- RL reward calculation
- A/B testing of models
- Dashboard analytics
