import base64
import tempfile
from subprocess import run
from tempfile import NamedTemporaryFile

import whisper
import torch
import torchaudio
from beam import App, Image, Runtime, Volume, VolumeType
from lang_list import (
    LANGUAGE_NAME_TO_CODE,
)

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 120  # in seconds

app = App(
    name="whisper",
    runtime=Runtime(
        cpu=4,
        memory="8Gi",
        gpu="T4",
        image=Image(
            python_packages=["git+https://github.com/openai/whisper.git"],
            commands=["apt-get update && apt-get install -y ffmpeg"],
        ),
    ),
    volumes=[Volume(path="./cache", name="cache_whisper", volume_type=VolumeType.Persistent)],
)

LANG_TO_ID = {v: k for k, v in whisper.tokenizer.LANGUAGES}


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(
        "medium", device=device, download_root="./cache"
    )
    return model


@app.rest_api(keep_warm_seconds=120, loader=load_model)
def transcribe_audio(**inputs):
    model = inputs["context"]

    # the bot gives languages in the SeamlessM4T format, so with initial capital letter
    target_language = inputs.get("target_language", "Italian").lower()
    task_name = inputs.get("task_name", "transcribe")
    
    if target_language not in LANG_TO_ID:
        return {"transcript": f"Target language {target_language} not supported."}

    # source_language_code = (
    # LANGUAGE_NAME_TO_CODE[source_language] if source_language else None
    # )
    # target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

    f1 = tempfile.NamedTemporaryFile()
    received_data = base64.b64decode(inputs["audio_file"].encode("utf-8"))
    f1.write(received_data)

    arr, org_sr = torchaudio.load(f1.name)

    print("Original SR:", org_sr)

    f1.close()
    new_arr = (
        torchaudio.functional.resample(
            arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE
        )
        if org_sr != AUDIO_SAMPLE_RATE
        else arr
    )

    # trim to max audio length
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        print(
            f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used."
        )

    f2 = tempfile.NamedTemporaryFile(suffix=".wav")
    torchaudio.save(f2.name, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))

    result = model.transcribe(
        f2.name,
        decode_options={"task": task_name, "language": target_language}
    )

    f2.close()
    return {"transcript": result["text"]}


if __name__ == "__main__":
    """'
    *** Testing Locally ***

    > beam start app.py
    > python app.py

    """
    import os

    mp3_filepath = os.path.abspath("example.ogg")
    transcribe_audio(
        audio_file=base64.b64encode(open(mp3_filepath, "rb").read()).decode("UTF-8"),
    )
