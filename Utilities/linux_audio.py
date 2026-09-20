"""Resolve the UI's PulseAudio descriptions to PortAudio's native endpoint names."""

def pulse_device_name(name, is_input=True):
    import pulsectl

    # A fresh connection reflects hotplug/profile changes. Never change the
    # desktop's default source/sink merely to select an application stream.
    with pulsectl.Pulse("whispering-tiger-device-map", connect=False) as pulse:
        pulse.connect(timeout=3)
        devices = pulse.source_list() if is_input else pulse.sink_list()
        return match_pulse_device_name(name, devices)


def match_pulse_device_name(name, devices):
    for device in devices:
        if device.name == name:
            return device.name
    matches = [device.name for device in devices if device.description == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple PulseAudio devices are named {name!r}; use a unique device description.")
    raise ValueError(f"PulseAudio device {name!r} is unavailable.")
