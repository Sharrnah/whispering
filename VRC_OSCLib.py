# Copyright (c) 2013 Iris
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

# Request "pythonosc" https://pypi.org/project/python-osc/

# OSC Imput Event Name => https://docs.vrchat.com/v2022.1.1/docs/osc-as-input-controller

import time
import os
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder
from unidecode import unidecode
from collections import deque
from itertools import count
import threading
import re

# Chatbox messages are queued as complete sequences.  This is intentionally
# separate from the other OSC helpers in this module: only /chatbox/input needs
# the FIFO/realtime replacement and rate-limit semantics below.
last_message_sent_time = 0.0
min_time_between_messages = 1.5
osc_chat_debug_logging = os.environ.get("ENABLE_LOGGING") == "1"

_message_ids = count(1)
_message_id_lock = threading.Lock()
_timing_lock = threading.Lock()
_chat_clients = {}
_chat_clients_lock = threading.Lock()


class _ChatSequenceQueue:
    """FIFO queue which may coalesce replaceable realtime sequences.

    Strict-FIFO final sequences are never removed.  A final sequence explicitly
    marked ``prioritize_latest`` cancels queued finals and the unsent remainder
    of the active final.  Enqueuing any newer chat sequence also invalidates
    pending/in-progress realtime data because the newer sequence is the more
    recent view of the chatbox.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._items = deque()
        self._unfinished_tasks = 0
        self._generation = 0
        self._latest_realtime_id = None
        self._active_sequence = None

    def put(self, sequence):
        with self._condition:
            previous_realtime_id = self._latest_realtime_id
            self._generation += 1
            sequence["generation"] = self._generation

            # A pending preview is obsolete as soon as any newer chatbox state
            # (another preview or the final transcript) is available.
            kept = deque()
            removed_ids = []
            skipped_final_ids = []
            for pending in self._items:
                if pending["replaceable"]:
                    removed_ids.append(pending["id"])
                    self._unfinished_tasks -= 1
                elif sequence["prioritize_latest"]:
                    pending["cancelled"] = True
                    skipped_final_ids.append(pending["id"])
                    self._unfinished_tasks -= 1
                else:
                    kept.append(pending)
            self._items = kept

            skipped_active_final_id = None
            if (
                sequence["prioritize_latest"]
                and self._active_sequence is not None
                and not self._active_sequence["replaceable"]
            ):
                self._active_sequence["cancelled"] = True
                skipped_active_final_id = self._active_sequence["id"]

            self._items.append(sequence)
            self._unfinished_tasks += 1
            self._latest_realtime_id = sequence["id"] if sequence["replaceable"] else None

            if previous_realtime_id is not None:
                replacement = (
                    sequence["id"]
                    if sequence["replaceable"]
                    else f"final id={sequence['id']}"
                )
                _chat_debug(f"replace realtime id={previous_realtime_id} -> {replacement}")
            for removed_id in removed_ids:
                if removed_id != previous_realtime_id:
                    _chat_debug(f"discard queued realtime id={removed_id}")
            if skipped_active_final_id is not None:
                _chat_debug(
                    f"supersede active final id={skipped_active_final_id} -> id={sequence['id']}"
                )
            for skipped_id in skipped_final_ids:
                _chat_debug(f"skip queued final id={skipped_id} -> id={sequence['id']}")
            total_chunks = len(sequence["chunks"])
            for chunk_index, message_data in enumerate(sequence["chunks"], start=1):
                _chat_debug(
                    f"enqueue id={sequence['id']} chunk={chunk_index}/{total_chunks} "
                    f"final={not sequence['replaceable']} "
                    f"utf16={count_utf16_code_units(message_data['data'])}"
                )

            self._condition.notify()

    def get(self):
        with self._condition:
            while not self._items:
                self._condition.wait()
            sequence = self._items.popleft()
            self._active_sequence = sequence
            return sequence

    def is_stale(self, sequence):
        with self._condition:
            if sequence["replaceable"]:
                return sequence["generation"] != self._generation
            return sequence["cancelled"]

    def has_pending_final(self):
        with self._condition:
            return any(not pending["replaceable"] for pending in self._items)

    def task_done(self, sequence):
        with self._condition:
            if self._unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            if self._active_sequence is sequence:
                self._active_sequence = None
            if self._latest_realtime_id == sequence["id"]:
                self._latest_realtime_id = None
            if self._unfinished_tasks == 0:
                self._condition.notify_all()

    def join(self):
        with self._condition:
            while self._unfinished_tasks:
                self._condition.wait()

    def empty(self):
        with self._condition:
            return not self._items

    def qsize(self):
        with self._condition:
            return len(self._items)


osc_queue = _ChatSequenceQueue()


def AV3_SetInt(data=0, Parameter="example", IP='127.0.0.1', PORT=9000):
    Int(data, "/avatar/parameters/" + Parameter, IP, PORT)


def AV3_SetFloat(data=0.0, Parameter="example", IP='127.0.0.1', PORT=9000):
    Float(data, "/avatar/parameters/" + Parameter, IP, PORT)


def AV3_SetBool(data=False, Parameter="example", IP='127.0.0.1', PORT=9000):
    Bool(data, "/avatar/parameters/" + Parameter, IP, PORT)


def Control_Push(Button="example", IP='127.0.0.1', PORT=9000):
    Buttons("/input/" + Button, IP, PORT)


def Control_Joystick(data=0.0, axis="example", IP='127.0.0.1', PORT=9000):
    Float(data, "/input/" + axis, IP, PORT)


def RemoveNonASCII(data):
    new_val = data.encode("ascii", "ignore")
    return new_val.decode()


# Button
def Buttons(address="/input/example", IP='127.0.0.1', PORT=9000):
    # OSC Bild
    client = udp_client.UDPClient(IP, PORT)
    msg = OscMessageBuilder(address=address)
    msg.add_arg(1)
    m = msg.build()

    msgb = OscMessageBuilder(address=address)
    msgb.add_arg(0)
    mb = msgb.build()

    # OSC Send
    client.send(m)
    time.sleep(0.1)
    client.send(mb)


# Int
def Int(data=0, address="/input/example", IP='127.0.0.1', PORT=9000):
    senddata = int(data)
    # OSC Bild
    client = udp_client.UDPClient(IP, PORT)
    msg = OscMessageBuilder(address=address)
    msg.add_arg(senddata)
    m = msg.build()

    # OSC Send
    client.send(m)


# Float
def Float(data=0.0, address="/input/example", IP='127.0.0.1', PORT=9000):
    senddata = float(data)
    # OSC Bild
    client = udp_client.UDPClient(IP, PORT)
    msg = OscMessageBuilder(address=address)
    msg.add_arg(senddata)
    m = msg.build()

    # OSC Send
    client.send(m)


# Bool
def Bool(data=False, address="/input/Jump", IP='127.0.0.1', PORT=9000):
    # OSC Bild
    client = udp_client.UDPClient(IP, PORT)
    msg = OscMessageBuilder(address=address)
    msg.add_arg(data)
    m = msg.build()

    # OSC Send
    client.send(m)


def set_min_time_between_messages(time_in_seconds):
    global min_time_between_messages
    interval = float(time_in_seconds)
    if interval < 0:
        raise ValueError("min_time_between_messages must not be negative")
    with _timing_lock:
        min_time_between_messages = interval


def set_chat_debug_logging(enabled):
    """Enable metadata-only logging for the OSC chatbox queue."""
    global osc_chat_debug_logging
    osc_chat_debug_logging = bool(enabled)


def _chat_debug(message):
    if osc_chat_debug_logging:
        print("[OSC CHAT] " + message)


def _next_message_id():
    with _message_id_lock:
        return next(_message_ids)


def _wait_for_send_slot(sequence, delay_before):
    """Wait for both the global rate limit and this sequence's chunk delay."""
    while True:
        if osc_queue.is_stale(sequence):
            return False

        with _timing_lock:
            last_sent = last_message_sent_time
            minimum_interval = min_time_between_messages

        # Long per-chunk delays are useful while one transcript is being read,
        # but should not make the next completed utterance wait 8-15 seconds per
        # remaining chunk.  Preserve FIFO and every final chunk, while draining
        # the current sequence at the safe global rate once another final waits.
        effective_chunk_delay = delay_before
        if not sequence["replaceable"] and osc_queue.has_pending_final():
            effective_chunk_delay = 0.0

        remaining = (
            last_sent
            + max(minimum_interval, effective_chunk_delay)
            - time.monotonic()
        )
        if remaining <= 0:
            return True

        # Short waits let a superseded realtime sequence yield quickly without
        # introducing a stop flag shared with final messages.
        time.sleep(min(remaining, 0.1))


def _send_osc_message():
    global last_message_sent_time
    while True:
        sequence = osc_queue.get()
        try:
            total_chunks = len(sequence["chunks"])
            for chunk_index, message_data in enumerate(sequence["chunks"], start=1):
                delay_before = message_data["delay_before"]
                send_data = {
                    key: value for key, value in message_data.items() if key != "delay_before"
                }
                if not _wait_for_send_slot(sequence, delay_before):
                    if sequence["replaceable"]:
                        _chat_debug(
                            f"discard realtime id={sequence['id']} "
                            f"remaining={total_chunks - chunk_index + 1}"
                        )
                    else:
                        _chat_debug(
                            f"stop superseded final id={sequence['id']} "
                            f"remaining={total_chunks - chunk_index + 1}"
                        )
                    break

                sent = False
                for attempt in range(1, 4):
                    if osc_queue.is_stale(sequence):
                        break
                    try:
                        _direct_osc_send(**send_data)
                        with _timing_lock:
                            last_message_sent_time = time.monotonic()
                        sent = True
                        _chat_debug(
                            f"send id={sequence['id']} chunk={chunk_index}/{total_chunks} "
                            f"final={not sequence['replaceable']}"
                        )
                        break
                    except Exception as e:
                        print(
                            f"[OSC CHAT] send error id={sequence['id']} "
                            f"chunk={chunk_index}/{total_chunks} attempt={attempt}/3: {e}"
                        )
                        time.sleep(0.1)

                if not sent:
                    if not osc_queue.is_stale(sequence):
                        print(
                            f"[OSC CHAT] sequence failed id={sequence['id']} "
                            f"at chunk={chunk_index}/{total_chunks}; remaining chunks were not sent"
                        )
                    break
        except Exception as e:
            print(f"[OSC CHAT] sender error id={sequence.get('id', '?')}: {e}")
        finally:
            osc_queue.task_done(sequence)


# OSC Send Command
def Message(data="example", address="/example", IP='127.0.0.1', PORT=9000):
    # OSC Bild
    client = udp_client.UDPClient(IP, PORT)
    msg = OscMessageBuilder(address=address)
    msg.add_arg(data)
    m = msg.build()

    # OSC Send
    client.send(m)


def _enqueue_chat_sequence(messages, replaceable=False, prioritize_latest=False):
    message_id = _next_message_id()
    sequence = {
        "id": message_id,
        "replaceable": bool(replaceable),
        "prioritize_latest": bool(prioritize_latest and not replaceable),
        "cancelled": False,
        "chunks": messages,
    }
    osc_queue.put(sequence)
    return message_id


def _chat_message_data(data, send, nofify, address, IP, PORT, convert_ascii, delay_before=0.0):
    return {
            "data": data,
            "send": send,
            "nofify": nofify,
            "address": address,
            "IP": IP,
            "PORT": PORT,
            "convert_ascii": convert_ascii,
            "delay_before": max(0.0, float(delay_before)),
        }


def Chat(data="example", send=True, nofify=True, address="/chatbox/input", IP='127.0.0.1', PORT=9000,
         convert_ascii=False, replaceable=False, prioritize_latest=False):
    """Queue one chatbox message.

    ``replaceable`` is intended for realtime transcription previews.  The
    default is deliberately reliable FIFO behavior for final/manual messages.
    """
    message = _chat_message_data(data, send, nofify, address, IP, PORT, convert_ascii)
    return _enqueue_chat_sequence(
        [message], replaceable=replaceable, prioritize_latest=prioritize_latest
    )


# OSC Send Chat
def _direct_osc_send(data="example", send=True, nofify=True, address="/chatbox/input", IP='127.0.0.1', PORT=9000, convert_ascii=False):
    # Reuse one socket per destination.  Besides avoiding a socket leak, a
    # stable source port removes needless variability from VRChat's UDP input.
    destination = (IP, int(PORT))
    with _chat_clients_lock:
        client = _chat_clients.get(destination)
        if client is None:
            client = udp_client.UDPClient(*destination)
            _chat_clients[destination] = client
            _chat_debug(f"udp client target={destination[0]}:{destination[1]}")

    msg = OscMessageBuilder(address=address)
    if convert_ascii:
        msg.add_arg(unidecode(data))
    else:
        msg.add_arg(data)
    msg.add_arg(send)
    msg.add_arg(nofify)
    m = msg.build()

    # OSC Send
    try:
        client.send(m)
    except Exception:
        # Let the worker retry with a fresh socket after a concrete socket error.
        with _chat_clients_lock:
            if _chat_clients.get(destination) is client:
                _chat_clients.pop(destination)
        client._sock.close()
        raise


def count_utf16_code_units(s):
    return len(s.encode('utf-16le')) // 2


def _split_by_utf16_limit(text, limit):
    """Hard-split text without cutting a Unicode code point/surrogate pair."""
    limit = int(limit)
    if limit <= 0:
        raise ValueError("chunk_size must be greater than zero")

    chunks = []
    current = []
    current_units = 0
    for character in text:
        character_units = count_utf16_code_units(character)
        if character_units > limit:
            raise ValueError(
                f"chunk_size={limit} cannot contain a {character_units}-unit Unicode character"
            )
        if current and current_units + character_units > limit:
            chunks.append("".join(current))
            current = []
            current_units = 0
        current.append(character)
        current_units += character_units

    if current:
        chunks.append("".join(current))
    return chunks


def split_words(text, chunk_size):
    return split_words_preserve_whitespace(text, chunk_size)


# preserve original whitespace (including line breaks) when chunking
def split_words_preserve_whitespace(text, chunk_size, reserved_overhead=8):
    """
    Split text into chunks preserving all original whitespace (line breaks, multiple spaces, tabs).
    reserved_overhead: UTF-16 units reserved for continuation markers
    ('... ' prefix + ' ...' suffix).
    """
    chunk_size = int(chunk_size)
    reserved_overhead = int(reserved_overhead)
    limit = chunk_size - reserved_overhead
    if limit <= 0:
        raise ValueError("chunk_size must be larger than reserved_overhead")

    tokens = re.split(r'(\s+)', text)  # keeps whitespace tokens
    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for tok in tokens:
        if tok == "":
            continue

        # Hard-split oversized words and whitespace runs by UTF-16 units.  Python
        # character slicing is not sufficient here because emoji count as two.
        if count_utf16_code_units(tok) > limit:
            flush()
            chunks.extend(_split_by_utf16_limit(tok, limit))
            continue

        if count_utf16_code_units(current + tok) <= limit:
            current += tok
        else:
            flush()
            current = tok

    flush()
    return [c for c in chunks if c]


def _marked_chunks(text, chunk_size):
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if count_utf16_code_units(text) <= chunk_size:
        return [text]

    # At very small configured limits the two four-unit markers leave no useful
    # payload.  Omitting markers is preferable to violating the hard limit.
    if chunk_size <= 8:
        return _split_by_utf16_limit(text, chunk_size)

    payloads = split_words_preserve_whitespace(text, chunk_size, reserved_overhead=8)
    marked = []
    for index, payload in enumerate(payloads):
        prefix = "... " if index else ""
        suffix = " ..." if index != len(payloads) - 1 else ""
        chunk = prefix + payload + suffix
        if count_utf16_code_units(chunk) > chunk_size:
            raise AssertionError("OSC chunk exceeds the configured UTF-16 limit")
        marked.append(chunk)
    return marked


def _take_utf16_prefix(text, start, limit):
    units = 0
    end = start
    while end < len(text):
        character_units = count_utf16_code_units(text[end])
        if units + character_units > limit:
            break
        units += character_units
        end += 1
    if end == start and start < len(text):
        raise ValueError(
            f"chunk_size={limit} cannot contain a "
            f"{count_utf16_code_units(text[start])}-unit Unicode character"
        )
    return text[start:end], end


def _scrolling_chunks(text, chunk_size, scroll_size):
    chunk_size = int(chunk_size)
    scroll_size = int(scroll_size)
    if chunk_size <= 0 or scroll_size <= 0:
        raise ValueError("chunk_size and scroll_size must be greater than zero")
    if count_utf16_code_units(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        chunk, end = _take_utf16_prefix(text, start, chunk_size)
        chunks.append(chunk)
        if end == len(text):
            break
        _, next_start = _take_utf16_prefix(text, start, scroll_size)
        start = next_start
    return chunks


def _queue_chunks(chunks, delay, initial_delay, nofify, address, ip, port, convert_ascii,
                  replaceable, prioritize_latest):
    messages = []
    for index, chunk in enumerate(chunks):
        delay_before = 0.0
        if index == 1:
            delay_before = initial_delay
        elif index > 1:
            delay_before = delay
        messages.append(
            _chat_message_data(
                chunk,
                True,
                nofify and index == 0,
                address,
                ip,
                port,
                convert_ascii,
                delay_before=delay_before,
            )
        )
    return _enqueue_chat_sequence(
        messages, replaceable=replaceable, prioritize_latest=prioritize_latest
    )


def Chat_chunks(data="example", chunk_size=144, delay=1., initial_delay=1., nofify=True,
                address="/chatbox/input", ip='127.0.0.1', port=9000, convert_ascii=False,
                replaceable=False, prioritize_latest=False):
    return send_chunks_v2(
        data, chunk_size, delay, initial_delay, nofify, address, ip, port, convert_ascii,
        replaceable, prioritize_latest
    )


def Chat_scrolling_chunks(data="example", chunk_size=144, delay=1., initial_delay=1., scroll_size=1,
                          nofify=True, address="/chatbox/input", ip='127.0.0.1', port=9000,
                          convert_ascii=False, replaceable=False, prioritize_latest=False):
    return send_scrolling_chunks(
        data, chunk_size, delay, initial_delay, scroll_size, nofify, address, ip, port,
        convert_ascii, replaceable, prioritize_latest
    )


def send_chunks(text, chunk_size=144, delay=1., initial_delay=1., nofify=True,
                address="/chatbox/input", ip='127.0.0.1', port=9000, convert_ascii=False,
                replaceable=False, prioritize_latest=False):
    chunks = _split_by_utf16_limit(text, chunk_size)
    return _queue_chunks(
        chunks, delay, initial_delay, nofify, address, ip, port, convert_ascii,
        replaceable, prioritize_latest
    )


def send_chunks_v2(text, chunk_size=144, delay=1., initial_delay=1., nofify=True,
                   address="/chatbox/input", ip='127.0.0.1', port=9000, convert_ascii=False,
                   replaceable=False, prioritize_latest=False):
    chunks = _marked_chunks(text, chunk_size)
    return _queue_chunks(
        chunks, delay, initial_delay, nofify, address, ip, port, convert_ascii,
        replaceable, prioritize_latest
    )


def send_scrolling_chunks(text, chunk_size=144, delay=1., initial_delay=1., scroll_size=1,
                          nofify=True, address="/chatbox/input", ip='127.0.0.1', port=9000,
                          convert_ascii=False, replaceable=False, prioritize_latest=False):
    chunks = _scrolling_chunks(text, chunk_size, scroll_size)
    return _queue_chunks(
        chunks, delay, initial_delay, nofify, address, ip, port, convert_ascii,
        replaceable, prioritize_latest
    )


# Exactly one persistent worker owns all /chatbox/input sends.  It is a daemon
# so application shutdown never waits on a sleeping OSC thread.
osc_sender_thread = threading.Thread(target=_send_osc_message, name="osc-chat-sender", daemon=True)
osc_sender_thread.start()
