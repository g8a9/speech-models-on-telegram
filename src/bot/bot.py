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


load_dotenv()

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
        for model in ["Whisper v3"]  #  ["SeamlessM4T", "Whisper"]
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


async def show_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = f"Target language: {context.user_data['language']}\nModel: {context.user_data.get('model', 'SeamlessM4T')}"

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
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

    user_model_choice = context.user_data.get("model", "Whisper v3")

    file_id = update.message.voice.file_id
    new_file = await context.bot.get_file(file_id)
    byte_data = await new_file.download_as_bytearray()

    # Send to Beam API
    encode_audio = base64.b64encode(byte_data).decode("UTF-8")

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
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
    client = AudioInference(
        model="whisper-v3",
        base_url="https://audio-prod.us-virginia-1.direct.fireworks.ai",
        api_key=os.environ.get("FIREWORKS_API_KEY"),
    )

    # response = r.json()
    # text_output = response.get("transcript", error_message)
    response = await client.transcribe_async(audio=encode_audio)
    text_output = response.text

    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_output)


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
    application.add_handler(
        CallbackQueryHandler(language_callback_query, pattern="language")
    )
    application.add_handler(CallbackQueryHandler(model_callback_query, pattern="model"))
    application.add_handler(MessageHandler(filters.ALL, get_audio_transcript))

    application.run_polling()
