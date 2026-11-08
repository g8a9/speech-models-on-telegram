import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError,
)
from google import genai
from google.genai import types
import os

logger = logging.getLogger(__name__)


class GeminiHelper:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(2))
    def __call__(self, prompt, **generation_kwargs):
        """Process a single image and prompt."""

        # img = PIL.Image.open(image_path)

        prompt = prompt.strip("\n")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=generation_kwargs.get("max_new_tokens", 256)
            ),
        )

        finish_reason = response.candidates[0].finish_reason

        text = (
            response.text
            if finish_reason == types.FinishReason.STOP
            else str(finish_reason)
        )

        return text
