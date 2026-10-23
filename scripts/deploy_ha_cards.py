"""Register (or update) the Weatherstat Lovelace cards in Home Assistant.

The cards live in ``ha/weatherstat-cards.js``. They are installed as an
inline dashboard resource — a ``data:`` URI holding the whole module — so
nothing has to be copied onto the HA box and every change gets a new URL
(no stale browser cache). Re-run after editing the file:

    just ha-cards            # create or update the resource
    just ha-cards --dry-run  # show what would change

Uses HA_URL / HA_TOKEN from ~/.weatherstat/.env (an admin token is needed
for lovelace/resources/*).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import websockets

from weatherstat._data_dir import load_env_file, resolve_data_dir

CARD_FILE = Path(__file__).resolve().parents[1] / "ha" / "weatherstat-cards.js"
MARKER = "WEATHERSTAT-CARDS"  # present in the module's console banner; identifies our resource
DATA_PREFIX = "data:text/javascript;base64,"


def _decode(url: str) -> str:
    if not url.startswith("data:"):
        return ""
    head, _, body = url.partition(",")
    try:
        return base64.b64decode(body).decode() if head.endswith(";base64") else body
    except ValueError:
        return ""


async def _deploy(dry_run: bool) -> int:
    load_env_file(resolve_data_dir() / ".env")
    ha_url = os.environ.get("HA_URL", "").rstrip("/")
    token = os.environ.get("HA_TOKEN", "")
    if not ha_url or not token:
        print("HA_URL / HA_TOKEN not set (~/.weatherstat/.env)", file=sys.stderr)
        return 1
    ws_url = ha_url.replace("https://", "wss://").replace("http://", "ws://") + "/api/websocket"
    content = CARD_FILE.read_text()
    url = DATA_PREFIX + base64.b64encode(content.encode()).decode()

    async with websockets.connect(ws_url, max_size=2**24) as ws:
        await ws.recv()  # auth_required
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        if json.loads(await ws.recv()).get("type") != "auth_ok":
            print("HA rejected the token", file=sys.stderr)
            return 1

        msg_id = 0

        async def call(payload: dict) -> dict:
            nonlocal msg_id
            msg_id += 1
            await ws.send(json.dumps({"id": msg_id, **payload}))
            while True:
                reply = json.loads(await ws.recv())
                if reply.get("id") == msg_id:
                    if not reply.get("success"):
                        raise RuntimeError(reply.get("error"))
                    return reply.get("result")

        resources = await call({"type": "lovelace/resources"})
        ours = [r for r in resources if MARKER in _decode(r.get("url", ""))]
        if ours and ours[0]["url"] == url:
            print(f"Up to date (resource {ours[0]['id']}, {len(content):,} bytes)")
            return 0
        if dry_run:
            print(f"Would {'update' if ours else 'create'} the resource ({len(content):,} bytes)")
            return 0
        if ours:
            await call({"type": "lovelace/resources/update", "resource_id": ours[0]["id"], "res_type": "module", "url": url})
            for extra in ours[1:]:
                await call({"type": "lovelace/resources/delete", "resource_id": extra["id"]})
            print(f"Updated resource {ours[0]['id']} ({len(content):,} bytes). Reload the dashboard to pick it up.")
        else:
            created = await call({"type": "lovelace/resources/create", "res_type": "module", "url": url})
            print(f"Created resource {created['id']} ({len(content):,} bytes). Reload the dashboard to pick it up.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    args = parser.parse_args()
    sys.exit(asyncio.run(_deploy(args.dry_run)))


if __name__ == "__main__":
    main()
