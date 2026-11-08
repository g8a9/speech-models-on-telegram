import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
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

load_dotenv()

from .gemini import GeminiClient
from .lang_list import LANGUAGE_CODE_TO_NAME, LANGUAGE_NAME_TO_CODE

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

MODE_TRANSCRIBE = "transcribe"
MODE_SUMMARIZE = "summarize"
MODE_LABELS = {
    MODE_TRANSCRIBE: "Transcribe",
    MODE_SUMMARIZE: "Summarize",
}
AUTO_LANGUAGE = "auto"
LANGUAGES_PER_ROW = 3

_gemini: Optional[GeminiClient] = None


def get_gemini() -> GeminiClient:
    """Create the Gemini client on first use, so importing the module needs no API key."""
    global _gemini
    if _gemini is None:
        _gemini = GeminiClient()
    return _gemini

welcome_message = """
Hi, this is Voice Bot. Send or forward me a voice note and I will turn it into text. Your voice note can be in any language: I detect it automatically.

Choose what you want:
- Transcribe: a clean, readable transcript.
- Summarize: a concise summary of what was said.

Use /mode to switch, /language to force a specific language, and /config to see your settings.
"""

error_message = """
There was an error while processing your voice note. It should be temporary, so try again in a while :)
"""


###
# Keyboard Markups and Callbacks
###
def get_mode_picker():
    keyboard = [
        [
            InlineKeyboardButton(label, callback_data=f"mode_{mode}")
            for mode, label in MODE_LABELS.items()
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_language_picker():
    buttons = [
        InlineKeyboardButton(
            "Auto-detect (default)", callback_data=f"language_{AUTO_LANGUAGE}"
        )
    ]
    buttons += [
        InlineKeyboardButton(name, callback_data=f"language_{code}")
        for name, code in LANGUAGE_NAME_TO_CODE.items()
    ]
    keyboard = [
        buttons[i : i + LANGUAGES_PER_ROW]
        for i in range(0, len(buttons), LANGUAGES_PER_ROW)
    ]
    return InlineKeyboardMarkup(keyboard)


async def mode_callback_query(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()
    choice = query.data.removeprefix("mode_")
    if choice not in MODE_LABELS:
        return

    context.user_data["mode"] = choice
    await query.edit_message_text(text=f"Selected mode: {MODE_LABELS[choice]}")


async def language_callback_query(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()
    choice = query.data.removeprefix("language_")

    if choice == AUTO_LANGUAGE:
        context.user_data.pop("language", None)
        await query.edit_message_text(text="Selected language: Auto-detect")
        return

    if choice not in LANGUAGE_CODE_TO_NAME:
        return

    context.user_data["language"] = choice
    await query.edit_message_text(
        text=f"Selected language: {LANGUAGE_CODE_TO_NAME[choice]}"
    )


###
# User preferences
###
def get_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    mode = context.user_data.get("mode", MODE_TRANSCRIBE)
    return mode if mode in MODE_LABELS else MODE_TRANSCRIBE


def get_forced_language(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Return the language code the user forced, or None for auto-detection."""
    code = context.user_data.get("language")
    return code if code in LANGUAGE_CODE_TO_NAME else None


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

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=welcome_message,
        reply_markup=get_mode_picker(),
    )


async def choose_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Select a mode. You can always change it with /mode",
        reply_markup=get_mode_picker(),
    )


async def choose_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=(
            "Select a language, or keep auto-detect to let the model understand it. "
            "You can always change it with /language"
        ),
        reply_markup=get_language_picker(),
    )


async def show_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = get_mode(context)
    code = get_forced_language(context)
    language = LANGUAGE_CODE_TO_NAME[code] if code else "Auto-detect"

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Mode: {MODE_LABELS[mode]}\nLanguage: {language}",
    )


###
# Voice notes
###
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    new_file = await context.bot.get_file(voice.file_id)
    audio_bytes = bytes(await new_file.download_as_bytearray())
    mime_type = voice.mime_type or "audio/ogg"

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    mode = get_mode(context)
    language_code = get_forced_language(context)
    logger.info(
        f"Processing voice note with mode={mode} "
        f"language={language_code or 'auto-detect'}"
    )

    gemini = get_gemini()
    try:
        if mode == MODE_SUMMARIZE:
            language_name = (
                LANGUAGE_CODE_TO_NAME[language_code] if language_code else None
            )
            text = await asyncio.to_thread(
                gemini.transcribe_and_summarize,
                audio_bytes,
                mime_type,
                language_name,
            )
        else:
            text = await asyncio.to_thread(
                gemini.transcribe, audio_bytes, mime_type, language_code
            )
    except Exception:
        logger.exception("Failed to process voice note")
        text = None

    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=text or error_message
    )


def main():
    token = os.environ.get("TELEGRAM_TOKEN")
    persistence_data = PicklePersistence(filepath="persistence.pkl")

    application = (
        ApplicationBuilder()
        .token(token)
        .persistence(persistence=persistence_data)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("mode", choose_mode))
    application.add_handler(CommandHandler("language", choose_language))
    application.add_handler(CommandHandler("config", show_config))
    application.add_handler(CallbackQueryHandler(mode_callback_query, pattern="^mode_"))
    application.add_handler(
        CallbackQueryHandler(language_callback_query, pattern="^language_")
    )
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    application.run_polling()


if __name__ == "__main__":
    main()
