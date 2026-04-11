"""Entry point for the Weatherstat TUI dashboard."""

import argparse

from weatherstat.tui.app import WeatherstatApp


def main() -> None:
    parser = argparse.ArgumentParser(prog="weatherstat-tui")
    parser.add_argument("--live", action="store_true", help="Start in live execution mode")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Enable embedded web frontend (HTTP control + status from a phone)",
    )
    parser.add_argument(
        "--web-host",
        default="0.0.0.0",
        help="Bind host for the web frontend (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8765,
        help="Bind port for the web frontend (default: 8765)",
    )
    args = parser.parse_args()

    app = WeatherstatApp(
        live=args.live,
        web_enabled=args.web,
        web_host=args.web_host,
        web_port=args.web_port,
    )
    app.run()


if __name__ == "__main__":
    main()
