"""Lightweight text segmentation helpers shared by TTS adapters.

The chunker is the Bark-inspired implementation historically used by
``chatterbox_tts.py``.  Keeping it free of model imports lets other adapters use
the same sentence/quote-aware behavior without importing the Chatterbox runtime.
"""

import random
import re


DEFAULT_VALID_ENDING_CHARS = '.;!?！。？！\n"'
VOICE_TAG_PATTERN = re.compile(r"^[\ufeff\u200b\s]*\[([^]]+)]\s*(.*)$")


def split_long_segment(segment, goal_length, custom_chars=","):
    """Split an overlong segment near a preferred separator or word boundary."""
    segments = []
    while len(segment) > goal_length:
        split_point = -1
        if custom_chars:
            split_points = [segment.rfind(char, 0, goal_length) for char in custom_chars]
            split_points = [point for point in split_points if point != -1]
            if split_points:
                split_point = max(split_points) + 1
        if split_point == -1:
            split_point = segment.rfind(" ", 0, goal_length)
            if split_point == -1:
                split_point = goal_length
        new_segment = segment[:split_point].strip()
        segment = segment[split_point:].strip()
        if new_segment:
            segments.append(new_segment)
    if segment:
        segments.append(segment)
    return segments


def chunk_text(
    text,
    goal_length,
    max_length=None,
    jitter=0,
    custom_split_chars=",",
    valid_ending_chars=DEFAULT_VALID_ENDING_CHARS,
):
    """Split text into sentence-aware, bounded character chunks.

    Sentence boundaries are preferred once the target size is reached.  The
    hard limit defaults to 130 percent of the target and falls back to commas,
    spaces, and finally a character boundary for unusually long sentences.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    goal_length = max(1, int(goal_length))
    if max_length is None:
        max_length = int(goal_length * 1.3)
    max_length = max(goal_length, int(max_length))
    jitter = max(0, int(jitter))
    valid_endings = valid_ending_chars + custom_split_chars

    if jitter:
        goal_length = random.randint(goal_length - jitter, goal_length + jitter)
        max_length = random.randint(max_length - jitter, max_length + jitter)

    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)

    chunks = []
    in_quote = False
    current = ""
    split_positions = []
    position = -1
    end_position = len(text) - 1

    def seek(delta):
        nonlocal position, in_quote, current
        backwards = delta < 0
        for _ in range(abs(delta)):
            if backwards:
                position -= 1
                current = current[:-1]
            else:
                position += 1
                current += text[position]
            if text[position] == '"':
                in_quote = not in_quote
        return text[position]

    def peek(delta):
        candidate = position + delta
        return text[candidate] if 0 <= candidate < end_position else ""

    def commit():
        nonlocal current, split_positions
        chunks.append(current)
        current = ""
        split_positions = []

    while position < end_position:
        character = seek(1)
        if len(current) >= max_length:
            if split_positions and len(current) > goal_length / 2:
                seek(-(position - split_positions[-1]))
            else:
                while character not in ";!?.\n " and position > 0 and len(current) > goal_length:
                    character = seek(-1)
            commit()
        elif not in_quote and (
            character in ";!?\n" or (character == "." and peek(1) in "\n ")
        ):
            while position < len(text) - 1 and len(current) < max_length and peek(1) in "!?.":
                character = seek(1)
            split_positions.append(position)
            if len(current) >= goal_length:
                commit()
        elif in_quote and peek(1) == '"' and peek(2) in "\n ":
            seek(2)
            split_positions.append(position)
    chunks.append(current)

    chunks = [segment.strip() for segment in chunks]
    chunks = [
        segment
        for segment in chunks
        if segment and not re.match(r"^[\s.,;:!?]*$", segment)
    ]

    index = 0
    while index < len(chunks):
        if chunks[index][-1] not in valid_endings:
            if custom_split_chars and any(char in custom_split_chars for char in chunks[index]):
                if index < len(chunks) - 1:
                    chunks[index] += " " + chunks[index + 1]
                    chunks.pop(index + 1)
                continue
        index += 1

    final_segments = []
    index = 0
    while index < len(chunks):
        current_segment = chunks[index]
        if index < len(chunks) - 1 and current_segment[-1] not in valid_endings:
            combined = current_segment + " " + chunks[index + 1]
            if len(combined) <= max_length:
                chunks[index] = combined
                chunks.pop(index + 1)
                continue
            if len(current_segment) > max_length:
                current_segment = split_long_segment(
                    current_segment, goal_length, custom_split_chars
                )
        elif len(current_segment) > max_length:
            current_segment = split_long_segment(
                current_segment, goal_length, custom_split_chars
            )
        if current_segment:
            if isinstance(current_segment, list):
                final_segments.extend(current_segment)
            else:
                final_segments.append(current_segment)
        index += 1
    return final_segments


def has_voice_tags(text):
    """Return whether text contains a bracketed tag at the start of a line."""
    if not isinstance(text, str):
        return False
    return re.search(r"(?m)^[\ufeff\u200b\s]*\[[^]]+]", text) is not None


def parse_voice_tagged_text(text):
    """Parse line-start ``[voice_name]`` tags into ordered voice sections.

    Untagged leading text belongs to ``main``. Inline brackets remain ordinary
    text, and following untagged lines continue using the current voice.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections = []
    current_voice = "main"
    buffer = []

    def flush():
        nonlocal buffer
        if buffer:
            content = "\n".join(buffer).strip()
            if content:
                sections.append((current_voice, content))
            buffer = []

    for raw_line in lines:
        line = raw_line.lstrip("\ufeff\u200b")
        match = VOICE_TAG_PATTERN.match(line)
        if match:
            flush()
            current_voice = match.group(1).strip()
            remainder = match.group(2)
            if remainder and remainder.strip():
                buffer.append(remainder.strip())
        elif line.strip():
            buffer.append(line.strip())
    flush()
    return sections
