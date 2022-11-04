# Copyright (c) 2013 Iris
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

# Request "pythonosc" https://pypi.org/project/python-osc/

# OSC Imput Event Name => https://docs.vrchat.com/v2022.1.1/docs/osc-as-input-controller

import time
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder
from unidecode import unidecode


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
def Chat(data="example", send=True, nofify=True, address="/chatbox/input", IP='127.0.0.1', PORT=9000, convert_ascii=True):
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
