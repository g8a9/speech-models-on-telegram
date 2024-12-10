import base64
import logging
import os
from functools import wraps

import requests
from dotenv import load_dotenv
from lang_list import S2TT_TARGET_LANGUAGE_NAMES
from requests.auth import HTTPBasicAuth
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PicklePersistence,
    filters,
)

from fireworks.client.audio import AudioInference
from io import BytesIO


load_dotenv()
from gemini import GeminiHelper

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")
beam_sm4t_endpoint = os.environ.get("BEAM_SM4T_ENDPOINT")
beam_whisper_endpoint = os.environ.get("BEAM_WHISPER_ENDPOINT")

welcome_message = """
Hi, this is Voice Bot. You can send or forward voice note to me: I will trascribe them into text. Your voice note can be in any language! 
Before we get started, I need to know which language I should use. Pick one below or send /language to choose.

Behind the scene, I use Meta's SeamlessM4T model.

Happy Transcribing.
"""
error_message = """
There was an error when trascribing your voice note. It should be temporary, so try again in while :)
"""


###
# Keyboard Markups and Callbacks
###
def get_language_picker():
    keyboard = [
        [InlineKeyboardButton(language, callback_data=f"language_{language}")]
        for language in ["English", "Italian", "Spanish"] + S2TT_TARGET_LANGUAGE_NAMES
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup


def get_model_picker():
    keyboard = [
        [InlineKeyboardButton(model, callback_data=f"model_{model}")]
        for model in ["Whisper v3", "Whisper v3 Turbo"]  #  ["SeamlessM4T", "Whisper"]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup


async def language_callback_query(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()
    choice = query.data.split("_")[-1]
    context.user_data["language"] = choice
    await query.edit_message_text(text=f"Selected language: {choice}")


async def model_callback_query(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()
    choice = query.data.split("_")[-1]
    context.user_data["model"] = choice
    await query.edit_message_text(text=f"Selected model: {choice}")


###
# Commands
###
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "unique_chat_count" not in context.bot_data:
        context.bot_data["unique_chat_count"] = 1
    else:
        context.bot_data["unique_chat_count"] += 1

    logger.info(
        f"New start. Unique chat count: {context.bot_data['unique_chat_count']}"
    )

    reply_markup = get_language_picker()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=welcome_message,
        reply_markup=reply_markup,
    )


async def choose_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_markup = get_language_picker()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Select a language. You can always change it with /language",
        reply_markup=reply_markup,
    )


async def choose_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_markup = get_model_picker()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Select a model. You can always change it with /model",
        reply_markup=reply_markup,
    )


async def toggle_clean_transcript(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current_clean_transcript = context.user_data.get("clean_transcript", False)
    context.user_data["clean_transcript"] = not current_clean_transcript
    emoji = "✅" if context.user_data["clean_transcript"] else "❌"
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Clean transcript: {emoji}",
    )


async def toggle_summarize_transcript(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    current_summarize_transcript = context.user_data.get("summarize_transcript", False)
    context.user_data["summarize_transcript"] = not current_summarize_transcript
    emoji = "✅" if context.user_data["summarize_transcript"] else "❌"
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Summarize transcript: {emoji}",
    )


async def show_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    clean_emoji = "✅" if context.user_data.get("clean_transcript", False) else "❌"
    summarize_emoji = (
        "✅" if context.user_data.get("summarize_transcript", False) else "❌"
    )
    text = f"""
Target language: {context.user_data['language']}
Model: {context.user_data.get('model', 'SeamlessM4T')}
Clean transcript: {clean_emoji}
Summarize transcript: {summarize_emoji}"""

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
    )


def get_clean_prompt(transcript: str, do_summarize: bool):
    return (
        f"""
Post-process this transcript of an audio recording. Clean it, add punctuation where needed, and make it more polished but do not change the meaning. Preserve the source language. Answer only with the post-processed text.
Transcript: {transcript}"""
        if not do_summarize
        else f"""
Post-process this transcript of an audio recording. Clean it, add punctuation where needed, and create a shorter, more concise version, but do not change the meaning. Preserve the source language. Answer only with the post-processed text.
Transcript: {transcript}"""
    )


###
# Misc
###
def send_typing_action(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    async def command_func(update, context, *args, **kwargs):
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        await func(update, context, *args, **kwargs)

    return command_func


async def get_audio_transcript(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "language" not in context.user_data:
        reply_markup = get_language_picker()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please, first select a language. You can always change it with /language",
            reply_markup=reply_markup,
        )
        return

    file_id = update.message.voice.file_id
    new_file = await context.bot.get_file(file_id)
    byte_data = await new_file.download_as_bytearray()

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    if context.user_data["model"] == "Whisper v3":
        model = "whisper-v3"
        base_url = "https://audio-prod.us-virginia-1.direct.fireworks.ai"
    else:
        model = "whisper-v3-turbo"
        base_url = "https://audio-turbo.us-virginia-1.direct.fireworks.ai"

    logger.info(f"Using model: {model}")
    client = AudioInference(
        model=model,
        base_url=base_url,
        api_key=os.environ.get("FIREWORKS_API_KEY"),
    )

    # # Send to Beam API
    # encode_audio = base64.b64encode(byte_data).decode("UTF-8")
    with BytesIO(byte_data) as audio_stream:
        response = await client.transcribe_async(
            audio=audio_stream, language=context.user_data["language"]
        )

        transcript = response.text

        do_clean_transcript = context.user_data.get("clean_transcript", False)
        do_summarize = context.user_data.get("summarize_transcript", False)

        if do_clean_transcript or do_summarize:

            client = GeminiHelper(model_name="gemini-1.5-flash")
            prompt = get_clean_prompt(transcript, do_summarize=do_summarize)

            transcript = client(
                prompt=prompt,
                max_new_tokens=512,
            )

    # data = {
    #     "audio_file": encode_audio,
    #     "target_language": context.user_data["language"],
    # }
    # headers = {
    #     "Accept": "*/*",
    #     "Connection": "keep-alive",
    #     "Content-Type": "application/json",
    # }
    # endpoint = (
    #     beam_sm4t_endpoint
    #     if user_model_choice == "SeamlessM4T"
    #     else beam_whisper_endpoint
    # )
    # r = requests.post(
    #     endpoint,
    #     auth=HTTPBasicAuth(client_id, client_secret),
    #     headers=headers,
    #     json=data,
    # )

    # response = r.json()
    # text_output = response.get("transcript", error_message)

    await context.bot.send_message(chat_id=update.effective_chat.id, text=transcript)


if __name__ == "__main__":
    TOKEN = os.environ.get("TELEGRAM_TOKEN")
    persistence_data = PicklePersistence(filepath="persistence.pkl")

    application = (
        ApplicationBuilder()
        .token(TOKEN)
        .persistence(persistence=persistence_data)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("language", choose_language))
    application.add_handler(CommandHandler("model", choose_model))
    application.add_handler(CommandHandler("config", show_config))
    application.add_handler(CommandHandler("toggle_clean", toggle_clean_transcript))
    application.add_handler(
        CommandHandler("toggle_summary", toggle_summarize_transcript)
    )
    application.add_handler(
        CallbackQueryHandler(language_callback_query, pattern="language")
    )
    application.add_handler(CallbackQueryHandler(model_callback_query, pattern="model"))
    application.add_handler(MessageHandler(filters.ALL, get_audio_transcript))

    application.run_polling()
