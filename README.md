# speech-models-on-telegram

A Telegram bot that turns voice notes into text, powered by Google's Gemini API. Send or forward a voice note and the bot will either transcribe it or summarize it. Voice notes can be in any language.

### Modes

- **Transcribe** (default): a clean, readable transcript. Uses `gemini-3.5-transcribe` in ["smart" mode](https://aistudio.google.com/docs/transcribe#transcription-modes), which removes filler words and false starts and adds punctuation and formatting.
- **Summarize**: a concise summary of what was said, produced in a single pass by `gemini-3.8-flash` (no separate transcript step).

### Language

By default the bot does not set a language: the model detects it automatically, including code-switching. Use `/language` to force a specific language (the bot then passes it explicitly to the API).

### Commands

- `/start` – show the welcome message and pick a mode
- `/mode` – choose between Transcribe and Summarize
- `/language` – choose a language, or keep auto-detect
- `/config` – show the current mode and language

### Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
# Install dependencies and create the virtual environment
uv sync

# Configure the bot
cp .env.example .env
# edit .env and set TELEGRAM_TOKEN and GOOGLE_API_KEY

# Run the bot
uv run voice-bot
```

Alternatively, run it as a module:

```bash
uv run python -m bot.bot
```

The project metadata and dependencies live in `pyproject.toml`; `uv.lock` pins the resolved versions.

### Privacy

The bot does not log, save, preprocess, or post-process any user data, except for each user's preferences (mode and language), which are persisted locally in `persistence.pkl`. Audio is sent to the Gemini API for processing.
