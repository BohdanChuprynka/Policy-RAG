import os


# Keep tests independent from developer-local .env files.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
