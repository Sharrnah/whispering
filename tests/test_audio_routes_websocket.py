import json
import unittest
from unittest import mock

import websocket


class _Server:
    def __init__(self):
        self.sent = []
        self.broadcasts = []

    async def send(self, client, message):
        self.sent.append((client, json.loads(message)))

    async def broadcast(self, message, exclude_client=None):
        self.broadcasts.append((json.loads(message), exclude_client))

    def broadcast_message(self, message, exclude_client=None):
        self.broadcasts.append((json.loads(message), exclude_client))


class AudioRoutesWebsocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_enabling_plugin_updates_explicit_main_route(self):
        class SubtitlePlugin:
            def on_enable(self):
                pass

        server = _Server()
        plugin = SubtitlePlugin()
        message = {
            "type": "setting_change",
            "name": "plugins",
            "value": {"SubtitlePlugin": True},
        }

        with mock.patch.object(websocket.Plugins, "plugins", [plugin]), \
                mock.patch.object(
                    websocket.settings,
                    "GetOption",
                    return_value={},
                ), mock.patch.object(
                    websocket.settings, "SetOption"
                ) as set_option, mock.patch.object(
                    websocket.settings.SETTINGS,
                    "get_all_settings",
                    return_value={"plugins": {"SubtitlePlugin": True}},
                ), mock.patch.object(
                    websocket.audio_routes,
                    "enable_plugins_for_main",
                    return_value=(True, ["SubtitlePlugin"]),
                ) as enable_for_main:
            await websocket.custom_message_handler(server, message, "client")

        enable_for_main.assert_called_once_with(["SubtitlePlugin"])
        self.assertIn(
            mock.call("main_audio_plugins", ["SubtitlePlugin"]),
            set_option.call_args_list,
        )

    async def test_successful_update_is_applied_before_it_is_persisted(self):
        server = _Server()
        selected = {
            "routes": [{"id": "game", "name": "Game audio"}],
            "main_audio_plugins": [],
        }
        message = {
            "type": "audio_routes_update",
            "value": {
                "request_id": "request-1",
                "routes": selected["routes"],
                "main_audio_plugins": [],
            },
        }

        with mock.patch.object(
            websocket.audio_routes, "apply_audio_routes", return_value=selected
        ) as apply_routes, mock.patch.object(websocket.settings, "SetOption") as set_option:
            await websocket.custom_message_handler(server, message, "client")

        apply_routes.assert_called_once_with(selected["routes"], [])
        self.assertEqual(
            set_option.call_args_list,
            [
                mock.call("additional_audio_routes", selected["routes"]),
                mock.call("main_audio_plugins", []),
            ],
        )
        self.assertTrue(server.sent[0][1]["data"]["success"])
        self.assertEqual(server.sent[0][1]["data"]["request_id"], "request-1")

    async def test_single_route_toggle_does_not_dirty_main_plugin_routing(self):
        server = _Server()
        selected = {
            "routes": [{"id": "game", "name": "Game audio", "enabled": False}],
            "main_audio_plugins": ["SubtitlePlugin"],
        }
        message = {
            "type": "audio_routes_update",
            "value": {
                "request_id": "toggle-1",
                "operation": "set_enabled",
                "route_id": "game",
                "enabled": False,
            },
        }

        with mock.patch.object(
            websocket.audio_routes,
            "set_audio_route_enabled",
            return_value=selected,
        ) as set_enabled, mock.patch.object(
            websocket.settings, "SetOption"
        ) as set_option:
            await websocket.custom_message_handler(server, message, "client")

        set_enabled.assert_called_once_with("game", False)
        set_option.assert_called_once_with(
            "additional_audio_routes", selected["routes"]
        )
        self.assertTrue(server.sent[0][1]["data"]["success"])

    async def test_plugin_routing_update_persists_both_allowlists(self):
        server = _Server()
        selected = {
            "routes": [{"id": "game", "plugins": ["SubtitlePlugin"]}],
            "main_audio_plugins": [],
        }
        route_plugins = {"game": ["SubtitlePlugin"]}
        message = {
            "type": "audio_routes_update",
            "value": {
                "request_id": "routing-1",
                "operation": "plugin_routing",
                "route_plugins": route_plugins,
                "main_audio_plugins": [],
            },
        }

        with mock.patch.object(
            websocket.audio_routes,
            "update_audio_route_plugin_routing",
            return_value=selected,
        ) as update_routing, mock.patch.object(
            websocket.settings, "SetOption"
        ) as set_option:
            await websocket.custom_message_handler(server, message, "client")

        update_routing.assert_called_once_with(route_plugins, [])
        self.assertEqual(
            set_option.call_args_list,
            [
                mock.call("additional_audio_routes", selected["routes"]),
                mock.call("main_audio_plugins", []),
            ],
        )

    async def test_failed_stream_update_does_not_persist(self):
        server = _Server()
        message = {
            "type": "audio_routes_update",
            "value": {"request_id": "request-2", "routes": []},
        }

        with mock.patch.object(
            websocket.audio_routes,
            "apply_audio_routes",
            side_effect=RuntimeError("stream failed"),
        ), mock.patch.object(websocket.settings, "SetOption") as set_option:
            await websocket.custom_message_handler(server, message, "client")

        set_option.assert_not_called()
        self.assertFalse(server.sent[0][1]["data"]["success"])
        self.assertIn("stream failed", server.sent[0][1]["data"]["error"])
        self.assertEqual(server.broadcasts, [])

    async def test_unknown_operation_cannot_replace_routes(self):
        server = _Server()
        message = {
            "type": "audio_routes_update",
            "value": {"request_id": "bad-op", "operation": "unknown"},
        }

        with mock.patch.object(
            websocket.audio_routes, "apply_audio_routes"
        ) as apply_routes, mock.patch.object(
            websocket.settings, "SetOption"
        ) as set_option:
            await websocket.custom_message_handler(server, message, "client")

        apply_routes.assert_not_called()
        set_option.assert_not_called()
        self.assertFalse(server.sent[0][1]["data"]["success"])
        self.assertIn("Unsupported audio route operation", server.sent[0][1]["data"]["error"])


if __name__ == "__main__":
    unittest.main()
