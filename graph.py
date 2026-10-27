import os
import time
from typing import Literal

import httpx
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain_core.globals import set_debug
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# DEBUG=1 in .env prints every graph step and everything sent to and received from the model
DEBUG = os.getenv("DEBUG") == "1"
set_debug(DEBUG)

# Triple quotes make a multi-line string: line breaks reach the model as they are.
SYSTEM_PROMPT = """You are an assistant for the card game Magic: The Gathering.

You have the tool search_card that looks up real card data on Scryfall.
For a question about a specific card, call search_card once for that card.
Pass the card's English name. If the user's spelling looks wrong,
pass your best guess of the correct name: the search also tolerates small typos.
Do not call the tool for questions about mechanics or rules.

When search_card returns a card, answer strictly in this format:
Card: <name, exactly as the tool gave it>
Mana cost: <mana cost, exactly as the tool gave it>
Type: <type, exactly as the tool gave it>
Power/Toughness: <power/toughness from the tool; omit this line if the tool gave none>
Text: <card text, copied exactly as the tool gave it>
What it does: <one or two simple sentences explaining the text above>
If the tool returned several faces of one card, repeat these lines for each face.

When search_card says the card was not found and lists similar names, answer only:
"I couldn't find that exact card. Similar cards: <the similar names, exactly as the tool listed them>"
When search_card finds no similar names either, answer only:
"I couldn't find that card. Please check the spelling of its name."
When search_card returns an error, do not call it again and answer only:
"I couldn't reach the card database. Please try again later."

If the question is about a mechanic or keyword, answer strictly in this format:
Mechanic: <name>
In short: <one sentence>
How it works: <two or three short sentences>
Example card: <a card you are sure has this mechanic, or "I don't know">

Rules:
- Never invent card data: card fields come only from search_card.
- If you are not sure about a mechanic, write "I don't know" in that field.
- Do not pick an example card just because its name contains the mechanic's name.
- No greetings, introductions or conclusions. Do not use markdown symbols like ** or #."""

# Scryfall requires two headers on every request:
# a User-Agent with the app name and version, and Accept
SCRYFALL_HEADERS = {"User-Agent": "mtg-bot/0.1", "Accept": "application/json"}


# ---------- Tool ----------

# The model reads the docstring below: it decides when and how to call the tool from it.
@tool
def search_card(name: str) -> str:
    """Look up a Magic: The Gathering card on Scryfall by its English name.
    The name may be partial or slightly misspelled.
    Returns real card data, or similar card names if there is no exact match."""
    # Fuzzy search: tolerates typos and partial names
    response = httpx.get(
        "https://api.scryfall.com/cards/named",
        params={"fuzzy": name},
        headers=SCRYFALL_HEADERS,
        timeout=10,
    )

    # The card wasn't found, or the name matches several cards at once:
    # ask Scryfall for similar names instead
    if response.status_code != 200:
        time.sleep(0.1)  # Scryfall asks for 50–100 ms between requests
        response = httpx.get(
            "https://api.scryfall.com/cards/autocomplete",
            params={"q": name},
            headers=SCRYFALL_HEADERS,
            timeout=10,
        )
        similar_names = response.json()["data"][:5]
        if similar_names:
            return "Card not found. Similar card names: " + ", ".join(similar_names)
        else:
            return "Card not found. No similar card names."

    card = response.json()

    # Double-faced and split cards keep their data per face, in card_faces
    if "card_faces" in card:
        faces = card["card_faces"]
    else:
        faces = [card]

    text = ""
    for face in faces:
        text = text + "Name: " + face.get("name", "") + "\n"
        text = text + "Mana cost: " + face.get("mana_cost", "") + "\n"
        text = text + "Type: " + face.get("type_line", "") + "\n"
        text = text + "Text: " + face.get("oracle_text", "") + "\n"
        if "power" in face:
            text = text + "Power/Toughness: " + face["power"] + "/" + face["toughness"] + "\n"
    return text.strip()


# One list of tools for the whole file: both the model and ToolNode get it
tools = [search_card]


# ---------- Models: created once, when the module is imported ----------

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
# bind_tools tells the model about the tools: their names, descriptions and arguments
# are sent to Ollama with every request, along with the messages
llm_with_tools = llm.bind_tools(tools)

whisper = WhisperModel("small", device="cpu", compute_type="int8")


# ---------- State schema ----------

class BotState(MessagesState):
    # Path to the downloaded voice message; None if the user sent text
    audio_path: str | None


# ---------- Router: sits on the edge from START and decides where to go ----------

def route_input(state: BotState) -> Literal["transcribe", "call_model"]:
    if state.get("audio_path"):
        return "transcribe"
    else:
        return "call_model"


# ---------- Nodes ----------

def transcribe(state: BotState) -> dict:
    segments, info = whisper.transcribe(state["audio_path"], language="en")

    # Glue the recognized pieces of text into one string
    text = ""
    for segment in segments:
        text = text + segment.text
    text = text.strip()

    # The text goes into the history as the user's message, and the path is reset
    # so the next text message doesn't come back here with a stale path
    return {"messages": [HumanMessage(content=text)], "audio_path": None}


async def call_model(state: BotState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    # The model either answers with text or asks to call a tool (tool_calls)
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


# ---------- Graph assembly ----------

builder = StateGraph(BotState)
builder.add_node("transcribe", transcribe)
builder.add_node("call_model", call_model)
# Prebuilt node that runs the tools the model asked for.
# handle_tool_errors=True: a tool error reaches the model as a message instead of crashing the graph
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

builder.add_conditional_edges(START, route_input)
builder.add_edge("transcribe", "call_model")
# The agent loop: tool_calls present -> go to "tools", none -> tools_condition leads to END
builder.add_conditional_edges("call_model", tools_condition)
# Hand the tool result back to the model so it can write the answer
builder.add_edge("tools", "call_model")

graph = builder.compile(checkpointer=InMemorySaver(), debug=DEBUG)
