import json

import websocket

LOADING_QUEUE = {}


def set_loading_state(key, value):
    LOADING_QUEUE[key] = value
    websocket.BroadcastMessage(json.dumps({"type": "loading_state", "data": LOADING_QUEUE}))


def get_loading_state(key):
    if key in LOADING_QUEUE:
        return LOADING_QUEUE[key]
    else:
        return None
