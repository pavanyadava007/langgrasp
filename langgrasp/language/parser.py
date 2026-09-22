"""Command parser: free text -> structured intent for the grounding and spatial-resolution stages.

Deliberately small and rule-based: the open-vocabulary grounder does the heavy lifting on the noun phrase,
the parser only extracts the action, a colour attribute if present, a spatial reference, and the phrase to
ground. Everything it cannot interpret is passed through verbatim as the grounding phrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ACTIONS = {
    "pick": ["pick", "pick up", "grab", "take", "grasp", "get", "lift", "fetch"],
    "place": ["place", "put", "drop", "move"],
}
COLOR_WORDS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black", "grey", "gray", "cyan", "magenta", "brown"]
SPATIAL = {
    "left": ["on the left", "left one", "leftmost", "to the left", "left"],
    "right": ["on the right", "right one", "rightmost", "to the right", "right"],
    "front": ["in front", "closest", "nearest", "front"],
    "back": ["at the back", "farthest", "furthest", "back one", "behind"],
}
GENERIC_NOUNS = {"one", "object", "thing", "item", "it", "that"}
FILLER = {"the", "a", "an", "please", "up", "now", "robot", "hey", "you", "me"}


@dataclass
class Intent:
    action: str = "pick"
    color: str | None = None
    noun: str | None = None  # head noun as spoken (may be a synonym; grounder resolves)
    spatial: str | None = None
    phrase: str = ""  # what is sent to the grounder
    raw: str = ""
    generic: bool = False  # "the blue one": no object noun, attribute-only
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_command(text: str) -> Intent:
    raw = text
    t = normalize(text)
    intent = Intent(raw=raw)
    # action
    for act, words in ACTIONS.items():
        for w in sorted(words, key=len, reverse=True):
            if re.search(rf"\b{w}\b", t):
                intent.action = act
                t = re.sub(rf"\b{w}\b", " ", t, count=1)
                break
        else:
            continue
        break
    # spatial reference (longest phrase first)
    for key, phrases in SPATIAL.items():
        for ph in sorted(phrases, key=len, reverse=True):
            if re.search(rf"\b{ph}\b", t):
                intent.spatial = key
                t = re.sub(rf"\b{ph}\b", " ", t, count=1)
                break
        if intent.spatial:
            break
    t = re.sub(r"\s+", " ", t).strip()
    # "can you" politeness
    t = re.sub(r"\bcan you\b", " ", t)
    words = [w for w in t.split() if w not in FILLER]
    color = next((w for w in words if w in COLOR_WORDS), None)
    if color:
        intent.color = "gray" if color == "grey" else color
    rest = [w for w in words if w not in COLOR_WORDS]
    # drop a trailing "on/in/into the tray" style destination if present
    if "tray" in rest or "bin" in rest:
        idx = min(i for i, w in enumerate(rest) if w in ("tray", "bin"))
        cut = idx - 1 if idx >= 1 and rest[idx - 1] in ("on", "in", "into", "to", "onto") else idx
        rest = rest[:cut]
    nouns = [w for w in rest if w not in ("on", "in", "into", "to", "onto", "of", "and")]
    if nouns and all(n in GENERIC_NOUNS for n in nouns):
        intent.generic = True
        intent.noun = None
    elif nouns:
        intent.noun = " ".join(n for n in nouns if n not in GENERIC_NOUNS) or None
    if intent.noun:
        intent.phrase = f"{intent.color} {intent.noun}".strip() if intent.color else intent.noun
    elif intent.color:
        intent.phrase = f"{intent.color} object"
        intent.generic = True
    else:
        intent.phrase = " ".join(nouns) or normalize(raw)
        intent.notes.append("no noun or colour found; grounding the whole command")
    return intent
