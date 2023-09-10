import base64
import tempfile
from subprocess import run
from tempfile import NamedTemporaryFile

import torch
import torchaudio
from beam import App, Image, Runtime, Volume, VolumeType
from lang_list import (
    LANGUAGE_NAME_TO_CODE,
)
from seamless_communication.models.inference import Translator

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 120  # in seconds

app = App(
    name="seamlessM4T",
    runtime=Runtime(
        cpu=4,
        memory="8Gi",
        gpu="T4",
        image=Image(
            python_packages=[
                "git+https://github.com/facebookresearch/seamless_communication.git"
            ],
            commands=[
                # "apt-get update && apt-get install -y autoconf autogen automake build-essential libasound2-dev libflac-dev libogg-dev libtool libvorbis-dev libopus-dev libmp3lame-dev libmpg123-dev pkg-config python",
                # "git clone https://github.com/libsndfile/libsndfile.git && cd libsndfile && autoreconf -vif && ./configure --enable-werror && make && make check",
            ],
        ),
    ),
    volumes=[Volume(path="./cache", name="cache", volume_type=VolumeType.Persistent)],
)


def load_model():
    torch.hub.set_dir("./cache")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize a Translator object with a multitask model, vocoder on the GPU.
    translator = Translator(
        "seamlessM4T_large",
        vocoder_name_or_card="vocoder_36langs",
        device=device,
    )

    return translator


@app.rest_api(keep_warm_seconds=120, loader=load_model)
def transcribe_audio(**inputs):
    translator = inputs["context"]

    target_language = inputs.get("target_language", "Italian")
    task_name = inputs.get("task_name", "asr")

    # source_language_code = (
    # LANGUAGE_NAME_TO_CODE[source_language] if source_language else None
    # )
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

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

    text_out, wav, sr = translator.predict(
        input=f2.name,
        task_str=task_name,
        tgt_lang=target_language_code,
        # src_lang=source_language_code,
        ngram_filtering=True,
    )

    f2.close()
    return {"transcript": str(text_out)}


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
