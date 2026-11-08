import base64
import os
from typing import Optional

from google import genai
from tenacity import retry, stop_after_attempt, wait_random_exponential

TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
SUMMARY_MODEL = "gemini-3.8-flash"

SUMMARY_PROMPT = """\
Listen to this audio recording and write a concise summary of what is said.
Preserve the language of the recording and do not translate it.
Answer only with the summary."""


class GeminiClient:
    """Thin wrapper around the Gemini API for transcription and summarization."""

    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(2))
    def transcribe(
        self, audio_bytes: bytes, mime_type: str, language_code: Optional[str] = None
    ):
        """Transcribe audio to text with smart transcription (cleaned up and formatted).

        If `language_code` is None the model detects the language automatically.
        """
        transcription_config = {
            "mode": "smart",
            "language_codes": [language_code] if language_code else [],
        }

        interaction = self.client.interactions.create(
            model=TRANSCRIBE_MODEL,
            input=[self._audio_content(audio_bytes, mime_type)],
            generation_config={"transcription_config": transcription_config},
        )

        return interaction.output_text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(2))
    def transcribe_and_summarize(
        self, audio_bytes: bytes, mime_type: str, language_name: Optional[str] = None
    ):
        """Transcribe and summarize audio in a single pass."""
        prompt = SUMMARY_PROMPT
        if language_name:
            prompt = (
                f"{prompt}\nThe recording is in {language_name}. "
                "Write the summary in that language."
            )

        interaction = self.client.interactions.create(
            model=SUMMARY_MODEL,
            input=[
                {"type": "text", "text": prompt},
                self._audio_content(audio_bytes, mime_type),
            ],
        )

        return interaction.output_text

    @staticmethod
    def _audio_content(audio_bytes: bytes, mime_type: str) -> dict:
        return {
            "type": "audio",
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
            "mime_type": mime_type,
        }
