import base64
import logging
import os
from functools import wraps

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    PicklePersistence,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    CallbackQueryHandler
)

from ..seamlessM4T.lang_list import S2TT_TARGET_LANGUAGE_NAMES

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
client_id = os.environ.get("CLIENT_ID") 
client_secret = os.environ.get("CLIENT_SECRET")
beam_endpoint = os.environ.get("BEAM_ENDPOINT")

welcome_message="""
Hi, this is Voice Bot. You can send or forward voice note to me: I will trascribe them into text. Your voice note can be in any language! 
Before we get started, I need to know which language I should use. Pick one below or send /language to choose.

Behind the scene, I use Meta's SeamlessM4T model.

Happy Transcribing.
"""


def get_language_picker():
    keyboard = [
        [InlineKeyboardButton(language, callback_data=language)]
        for language in ["English", "Italian", "Spanish"] + S2TT_TARGET_LANGUAGE_NAMES
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "unique_chat_count" not in context.bot_data:
        context.bot_data["unique_chat_count"] = 1
    else:
        context.bot_data["unique_chat_count"] += 1
    
    logger.info(f"New start. Unique chat count: {context.bot_data['unique_chat_count']}")

    reply_markup = get_language_picker()
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=welcome_message, reply_markup=reply_markup
    )


async def language_button_pressed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()
    context.user_data["language"] = query.data
    await query.edit_message_text(text=f"Selected language: {query.data}")


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
            reply_markup=reply_markup
        )

    file_id = update.message.voice.file_id
    new_file = await context.bot.get_file(file_id)
    byte_data = await new_file.download_as_bytearray()
    
    encode_audio = base64.b64encode(byte_data).decode("UTF-8") 

    data = {
        "audio_file": encode_audio,
    }
    headers = {
        "Accept": "*/*",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
    }

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    r = requests.post(
        beam_endpoint,
        auth=HTTPBasicAuth(client_id, client_secret),
        headers=headers,
        json=data,
    )

    text_output = r.json()["transcript"]

    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_output)

async def choose_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    send_language_picker()

if __name__ == "__main__":
    TOKEN = os.environ.get("TELEGRAM_TOKEN")
    persistence_data = PicklePersistence(filepath='persistence.pkl')

    application = ApplicationBuilder().token(TOKEN).persistence(persistence=persistence_data).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("language", choose_language))
    application.add_handler(CallbackQueryHandler(language_button_pressed))
    application.add_handler(MessageHandler(filters.ALL, get_audio_transcript))

    application.run_polling()
