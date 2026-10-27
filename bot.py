# Telegram bot: takes a question as text or a voice message, passes it to the graph
# from graph.py and sends the model's answer back to the chat.

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from graph import graph

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

dp = Dispatcher()


@dp.message(CommandStart())
async def handle_start(message: Message):
    await message.answer(
        "Hi! Ask me anything about Magic: The Gathering — a mechanic or a specific card. "
        "You can type your question or send a voice message."
    )


@dp.message(F.text | F.voice)
async def handle_message(message: Message):
    # Placeholder message while the model is thinking. message.answer returns
    # the sent message, so we keep it to delete it at the end
    thinking = await message.answer("Thinking...")
    # Telegram shows "typing..." for 5 seconds at most; after that only the placeholder stays
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    # The handler only turns a Telegram message into graph input;
    # the graph decides where to route it (see route_input)
    audio_path = None
    if message.voice:
        # message_id is unique only within a chat, so the file name includes chat.id too
        audio_path = f"voice_{message.chat.id}_{message.message_id}.ogg"
        await message.bot.download(message.voice, destination=audio_path)
        graph_input = {"audio_path": audio_path}
    else:
        graph_input = {"messages": [HumanMessage(content=message.text)]}

    # recursion_limit caps the number of graph steps per call (like Max Iterations in n8n),
    # so the model can't loop through the tool forever
    config = {"configurable": {"thread_id": str(message.chat.id)}, "recursion_limit": 10}
    result = await graph.ainvoke(graph_input, config)

    # The voice file is no longer needed
    if audio_path:
        os.remove(audio_path)

    answer = result["messages"][-1].content
    if not answer:
        answer = "The model returned an empty answer. Please try rephrasing your question."

    # For a voice message, show what Whisper heard: it's the user's last message
    # in the history. [-2] won't do, since a tool call and its result
    # may sit between the question and the answer
    if audio_path:
        heard = ""
        for msg in result["messages"]:
            if msg.type == "human":
                heard = msg.content
        answer = f'Heard: "{heard}"\n\n{answer}'

    # The placeholder is no longer needed: delete it and send the real answer
    # as a new message, so the user gets a notification
    await thinking.delete()
    # Telegram limits a message to 4096 characters, so we cut with a margin
    await message.answer(answer[:4000])


async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    try:
        # On start, aiogram requests the bot's info from Telegram (get_me) once and doesn't
        # retry: if the network blinks at that moment, the bot crashes. So we make this
        # request ourselves, with retries. bot.me() caches the answer, so start_polling
        # won't go to the network for it again
        while True:
            try:
                await bot.me()
                break
            except TelegramNetworkError as error:
                logging.warning("Telegram is unreachable: %s. Retrying in 5 seconds", error)
                await asyncio.sleep(5)

        # From here on, aiogram reconnects by itself on network errors
        await dp.start_polling(bot)
    finally:
        # Close the connection to Telegram on any exit, including Ctrl+C before start
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Ctrl+C while the bot waits for Telegram: exit without a traceback
        pass
