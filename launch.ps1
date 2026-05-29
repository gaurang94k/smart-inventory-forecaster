$env:ANTHROPIC_API_KEY = "YOUR_KEY_HERE"
uv run uvicorn api.main:app --reload --port 8000
