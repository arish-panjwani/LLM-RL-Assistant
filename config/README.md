# Configuration

Contains API keys, environment variables, and project settings.

## Files
- `config.yaml` – Stores constants like RL weights, thresholds, model routes
- `secrets.env` – Groq API key, Google API, Wi-Fi settings, etc.

## Notes
- `secrets.env` should be excluded from Git using `.gitignore`
- Load secrets with `python-dotenv` or similar tools
