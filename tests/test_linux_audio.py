from types import SimpleNamespace
import unittest

from Utilities.linux_audio import match_pulse_device_name


class PulseAudioNameTests(unittest.TestCase):
    def test_display_name_resolves_to_native_endpoint(self):
        devices = [SimpleNamespace(name="alsa_input.usb-mic", description="USB Microphone")]
        self.assertEqual(match_pulse_device_name("USB Microphone", devices), "alsa_input.usb-mic")
        self.assertEqual(match_pulse_device_name("alsa_input.usb-mic", devices), "alsa_input.usb-mic")

    def test_monitor_is_preserved(self):
        devices = [SimpleNamespace(name="speakers.monitor", description="Monitor of Speakers")]
        self.assertEqual(match_pulse_device_name("Monitor of Speakers", devices), "speakers.monitor")

    def test_missing_or_ambiguous_device_does_not_select_another_microphone(self):
        with self.assertRaises(ValueError):
            match_pulse_device_name("Missing", [])
        devices = [SimpleNamespace(name=name, description="USB Mic") for name in ("a", "b")]
        with self.assertRaises(ValueError):
            match_pulse_device_name("USB Mic", devices)
