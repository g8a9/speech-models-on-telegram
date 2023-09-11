# speech-models-on-telegram

Code for serving Meta's SeamlessM4T via [Beam](https://www.beam.cloud/) and for running [Voice Bot](https://telegram.me/the_whisper_bot), a telegram bot to transform your voice notes into text.  

### Components

- A Beam App to serve a serverless inference REST endpoint for speech models. Roughly, it receives a voice note's bytes and returns a transcript text.
- A Python Bot to let people send or forward voice notes and forward them in turn to the Beam App.

The app does not log, save, preprocess, or post process any user data, except for each user's preference of preferred language.

### Models

[SeamlessM4T](https://ai.meta.com/blog/seamless-m4t/) is a multilingual and multitask model that translates and transcribes across speech and text. The goal of this project is to be able to serve it via an async Rest API and let an external client query it with audio files and get a response in textual format.
My ideal use case for such a client is a Telegram Bot.

**Useful Links**

- SeamlessM4T Demo on HF: https://huggingface.co/spaces/facebook/seamless_m4t 
- OpenAI's Whisper Demo on Beam: https://github.com/slai-labs/get-beam/blob/main/examples/whisper-tutorial/app.py

### Limitations

- We trim voice notes to a maximum of 60 seconds
- App gets suspended if not invoked for longer that 2 minutes. If that happens, then you'll cold start it and have to wait ~30 seconds to get your transcript.
- This is a side project, so
    - code is not nice and tidy
    - I can't guarantee 24/7 assistance
    - I can't guarantee it'll be up forever