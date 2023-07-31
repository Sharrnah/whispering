# Copyright (c) 2013 Iris
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

# Request "pythonosc" https://pypi.org/project/python-osc/

# OSC Imput Event Name => https://docs.vrchat.com/v2022.1.1/docs/osc-as-input-controller

import time
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder
from unidecode import unidecode
import threading

stop_flag = False
thread = None


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


# OSC Send Command
def Message(data="example", address="/example", IP='127.0.0.1', PORT=9000):
    # OSC Bild
    client = udp_client.UDPClient(IP, PORT)
    msg = OscMessageBuilder(address=address)
    msg.add_arg(data)
    m = msg.build()

    # OSC Send
    client.send(m)


# OSC Send Chat
def Chat(data="example", send=True, nofify=True, address="/chatbox/input", IP='127.0.0.1', PORT=9000, convert_ascii=False):
    # OSC Bild
    client = udp_client.UDPClient(IP, PORT)
    msg = OscMessageBuilder(address=address)
    if convert_ascii:
        msg.add_arg(unidecode(data))
    else:
        msg.add_arg(data)
    msg.add_arg(send)
    msg.add_arg(nofify)
    m = msg.build()

    # OSC Send
    client.send(m)


def Chat_chunks(data="example", chunk_size=144, delay=1., initial_delay=1., nofify=True, address="/chatbox/input", ip='127.0.0.1', port=9000, convert_ascii=False):
    global thread, stop_flag

    # stop thread
    if thread and thread.is_alive():
        stop_flag = True
        thread.join()

    stop_flag = False

    thread = threading.Thread(target=send_chunks_v2, args=(data, chunk_size, delay, initial_delay, nofify, address, ip, port, convert_ascii))
    thread.start()


def Chat_scrolling_chunks(data="example", chunk_size=144, delay=1., initial_delay=1., scroll_size=1, nofify=True, address="/chatbox/input", ip='127.0.0.1', port=9000, convert_ascii=False):
    global thread, stop_flag

    # stop thread
    if thread and thread.is_alive():
        stop_flag = True
        thread.join()

    stop_flag = False

    thread = threading.Thread(target=send_scrolling_chunks, args=(data, chunk_size, delay, initial_delay, scroll_size, nofify, address, ip, port, convert_ascii))
    thread.start()


# send chat by chunks
def send_chunks(text, chunk_size=144, delay=1., initial_delay=1., nofify=True, address="/chatbox/input", ip='127.0.0.1', port=9000, convert_ascii=False):
    # Convert text to list of UTF-16 code units
    text_utf16 = text.encode('utf-16le')

    # Check if text is shorter than chunk_size
    if len(text_utf16) <= chunk_size * 2:
        Chat(text, send=True, nofify=nofify, address=address, IP=ip, PORT=port, convert_ascii=convert_ascii)
        return

    # Calculate the number of chunks
    num_chunks = len(text_utf16) // (chunk_size * 2) + (len(text_utf16) % (chunk_size * 2) != 0)

    for i in range(num_chunks):
        if stop_flag:
            break

        # Get the current chunk
        chunk_utf16 = text_utf16[i*chunk_size*2:(i+1)*chunk_size*2]

        # Convert chunk back to string
        chunk = chunk_utf16.decode('utf-16le')

        # Send the chunk to the API
        Chat(chunk, send=True, nofify=(nofify and i == 0), address=address, IP=ip, PORT=port, convert_ascii=convert_ascii)

        # Wait for the specified delay
        if i == 0:
            time.sleep(initial_delay)
        else:
            time.sleep(delay)


def split_words(text, chunk_size):
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        # If a word is longer than the chunk size, split it into parts
        while len(word.encode('utf-16')) > (chunk_size - 6)*2:
            part, word = word[:(chunk_size - 6)//2], word[(chunk_size - 6)//2:]
            chunks.append(current_chunk + " " + part if current_chunk else part)
            current_chunk = ""

        # Adding 2 to account for the space that will be added, and 6 for the dots, all converted to UTF-16
        if len((current_chunk + word + " ...").encode('utf-16')) <= chunk_size*2:
            # Add the word to the current chunk
            if current_chunk != "":
                current_chunk += " "
            current_chunk += word
        else:
            # The current word would make the chunk too long, so it's time to start a new chunk
            chunks.append(current_chunk)
            current_chunk = word

    # Don't forget to add the last chunk
    if current_chunk != "":
        chunks.append(current_chunk)

    return chunks


def send_chunks_v2(text, chunk_size=144, delay=1., initial_delay=1., nofify=True, address="/chatbox/input", ip='127.0.0.1', port=9000, convert_ascii=False):
    # Send the full text if it fits into a single chunk
    if len(text.encode('utf-16')) <= chunk_size*2:
        Chat(text, send=True, nofify=nofify, address=address, IP=ip, PORT=port, convert_ascii=convert_ascii)
        return

    chunks = split_words(text, chunk_size)
    for i, chunk in enumerate(chunks):
        if stop_flag:
            break

        # Add dots to indicate continuation
        if i != 0:
            chunk = "..." + chunk
        if i != len(chunks) - 1:
            chunk = chunk + "..."

        # Send the chunk to the API
        Chat(chunk, send=True, nofify=(nofify and i == 0), address=address, IP=ip, PORT=port, convert_ascii=convert_ascii)

        # Wait for the specified delay
        if i == 0:
            time.sleep(initial_delay)
        else:
            time.sleep(delay)


# send chat by scrolling chunks
def send_scrolling_chunks(text, chunk_size=144, delay=1., initial_delay=1., scroll_size=1, nofify=True, address="/chatbox/input", ip='127.0.0.1', port=9000, convert_ascii=False):
    # Convert text to list of UTF-16 code units
    text_utf16 = text.encode('utf-16le')

    # Check if text is shorter than chunk_size
    if len(text_utf16) <= chunk_size * 2:
        Chat(text, send=True, nofify=nofify, address=address, IP=ip, PORT=port, convert_ascii=convert_ascii)
        return

    # Calculate the number of chunks
    num_chunks = (len(text_utf16) - chunk_size * 2) // (scroll_size * 2) + 1

    for i in range(num_chunks):
        if stop_flag:
            break

        # Get the current chunk
        chunk_utf16 = text_utf16[i*scroll_size*2:(i*scroll_size*2)+chunk_size*2]

        # Convert chunk back to string
        chunk = chunk_utf16.decode('utf-16le')

        # Send the chunk to the API
        Chat(chunk, send=True, nofify=(nofify and i == 0), address=address, IP=ip, PORT=port, convert_ascii=convert_ascii)

        # Wait for the specified delay
        if i == 0:
            time.sleep(initial_delay)
        else:
            time.sleep(delay)

