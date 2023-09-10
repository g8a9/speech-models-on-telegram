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
    PicklePersistence
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
Before we get started, I need to know which language I should use: Send /language to choose.

Behind the scene, I use Meta's SeamlessM4T model.

Happy Transcribing.
"""

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=welcome_message
    )


def send_typing_action(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    async def command_func(update, context, *args, **kwargs):
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        await func(update, context, *args, **kwargs)

    return command_func

def user_has_no_language():
    pass

def send_language_picker():
    supported_languages = S2TT_TARGET_LANGUAGE_NAMES
    return


async def get_audio_transcript(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if user_has_no_language():
        send_language_picker()
        return

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
    application.add_handler(MessageHandler(filters.ALL, get_audio_transcript))

    application.run_polling()
