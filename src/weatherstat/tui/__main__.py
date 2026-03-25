"""Entry point for the Weatherstat TUI dashboard."""

import sys

from weatherstat.tui.app import WeatherstatApp

live = "--live" in sys.argv[1:]
app = WeatherstatApp(live=live)
app.run()
