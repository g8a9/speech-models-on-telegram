import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError,
)
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)


class GeminiHelper:
    def __init__(self, model_name: str):
        self.model_name = model_name
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_name=model_name)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def __call__(self, prompt, **generation_kwargs):
        """Process a single image and prompt."""

        # img = PIL.Image.open(image_path)

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=generation_kwargs.get("max_new_tokens", 256)
            ),
        )

        finish_reason = response.candidates[0].finish_reason

        text = (
            response.text
            if finish_reason == genai.protos.Candidate.FinishReason.STOP
            else str(finish_reason)
        )

        return text
