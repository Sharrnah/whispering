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
    async def test_audio_attach_requires_secret_and_matching_peer(self):
        from types import SimpleNamespace
        import remote_audio
        session = SimpleNamespace(attach_token="private", control=None, socket=SimpleNamespace(remote_address=("192.168.1.25", 5001)))
        client = SimpleNamespace(remote_address=("192.168.1.26", 10000))
        with mock.patch.object(remote_audio, "host_server", SimpleNamespace(session=session)):
            await websocket.custom_message_handler(_Server(), {"type": "remote_audio_attach", "value": {"token": "private"}}, client)
            self.assertIsNone(session.control)
            client.remote_address = ("192.168.1.25", 10000)
            await websocket.custom_message_handler(_Server(), {"type": "remote_audio_attach", "value": {"token": "wrong"}}, client)
            self.assertIsNone(session.control)
            await websocket.custom_message_handler(_Server(), {"type": "remote_audio_attach", "value": {"token": "private"}}, client)
            self.assertIs(session.control, client)

    async def test_remote_languages_follow_profile_and_freeze_at_queue_boundary(self):
        import remote_audio
        import audio_routes
        import settings
        from types import SimpleNamespace
        import threading
        values = dict(settings.SETTINGS.get_all_settings())
        values.update(current_language="en", src_lang="en", trg_lang="de", txt_translate=True)
        captured = []
        session = SimpleNamespace(follow_profile=True, closed=threading.Event())
        with mock.patch.object(settings.SETTINGS, "GetOption", side_effect=lambda name: values.get(name)), mock.patch.object(settings.SETTINGS, "get_all_settings", side_effect=lambda: dict(values)), mock.patch.object(audio_routes.AudioRoute, "start", autospec=True, side_effect=lambda route, capture: captured.append(route)):
            remote_audio.create_source(session, {})
            route = captured[0]
            snapshot = route.settings.snapshot()
            values.update(current_language="ja", src_lang="ja", trg_lang="fr", txt_translate=False)
            self.assertEqual(route.settings.GetOption("current_language"), "ja")
            self.assertEqual(route.settings.GetOption("trg_lang"), "fr")
            self.assertFalse(route.settings.GetOption("txt_translate"))
            self.assertEqual(snapshot.GetOption("current_language"), "en")
            self.assertEqual(snapshot.GetOption("trg_lang"), "de")
            self.assertEqual(route.settings.GetOption("websocket_ip"), "0")

    async def test_remote_audio_administration_is_loopback_only(self):
        from types import SimpleNamespace
        import remote_audio
        server = _Server()
        with mock.patch.object(remote_audio, "configure_host", new_callable=mock.AsyncMock) as configure:
            await websocket.custom_message_handler(
                server, {"type": "remote_audio_host", "value": {"enabled": True}},
                SimpleNamespace(remote_address=("192.168.1.25", 10000)))
            configure.assert_not_awaited()
            self.assertEqual(server.sent, [])

    async def test_pairing_key_is_returned_only_to_local_requester(self):
        from types import SimpleNamespace
        import remote_audio
        server = _Server()
        client = SimpleNamespace(remote_address=("127.0.0.1", 10000))
        with mock.patch.object(remote_audio, "configure_host", new_callable=mock.AsyncMock,
                               return_value={"type": "remote_audio_host", "token": "secret"}) as configure:
            await websocket.custom_message_handler(
                server, {"type": "remote_audio_host", "value": {"enabled": True}}, client)
            configure.assert_awaited_once_with(True)
            self.assertEqual(server.sent, [(client, {"type": "remote_audio_host", "token": "secret"})])
            self.assertEqual(server.broadcasts, [])

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
