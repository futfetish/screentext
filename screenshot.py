from gi.repository import Gio, GLib
from urllib.parse import urlparse, unquote


def uri_to_path(uri):
    return unquote(urlparse(uri).path)


def take_screenshot():
    bus = Gio.bus_get_sync(
        Gio.BusType.SESSION,
        None
    )

    result = bus.call_sync(
        "org.freedesktop.portal.Desktop",
        "/org/freedesktop/portal/desktop",
        "org.freedesktop.portal.Screenshot",
        "Screenshot",
        GLib.Variant(
            "(sa{sv})",
            (
                "",
                {
                    "interactive": GLib.Variant("b", True)
                }
            )
        ),
        None,
        Gio.DBusCallFlags.NONE,
        -1,
        None
    )

    handle = result.unpack()[0]

    loop = GLib.MainLoop()

    screenshot_path = None

    def on_response(*args):
        nonlocal screenshot_path

        parameters = args[-1]

        response, results = parameters.unpack()

        if response == 0:
            screenshot_path = uri_to_path(
                results["uri"]
            )

        loop.quit()

    bus.signal_subscribe(
        None,
        "org.freedesktop.portal.Request",
        "Response",
        handle,
        None,
        Gio.DBusSignalFlags.NONE,
        on_response
    )

    loop.run()

    return screenshot_path