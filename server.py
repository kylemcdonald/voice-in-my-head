import sys
import os
import requests
import subprocess
import time
import atexit

from daily_helpers import delete_completed, room_url_to_name, delete_room

import requests
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from dotenv import load_dotenv
from flask_autoindex import AutoIndex
from urllib.parse import parse_qs


load_dotenv()
daily_api_key = os.getenv("DAILY_API_KEY")


def kill_subprocesses():
    print("killing subprocesses")
    for room_url, proc in subprocesses:
        room_name = room_url_to_name(room_url)
        print(room_name)
        proc.kill()
        proc.wait()
        delete_room(daily_api_key, room_name)


subprocesses = []
atexit.register(kill_subprocesses)

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    ip_addr = request.headers.get("X-Forwarded-For")
    print(
        f"page loaded from {ip_addr} {request.user_agent} {request.query_string.decode()}"
    )
    return send_from_directory("static-pages", "index.html")


@app.route("/sound/<path:path>")
def send_sound(path):
    return send_from_directory("sound", path)


@app.route("/style.css")
def stylesheet():
    return send_from_directory("static-pages", "style.css")


@app.route("/delete-completed")
def delete_completed_route():
    return delete_completed(daily_api_key)


@app.route("/run")
def run():
    return send_from_directory("static-pages", "run.html")


@app.route("/update-script")
def update_script():
    subprocess.run(["bash", "update-script.sh"])
    with open("scripts/script.csv", "r") as f:
        script = f.read()
    return script, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/info")
def info():
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    return commit + " " + os.getenv("LOCATION")


def convert_dict_to_args(d):
    args = []
    for k, v in d.items():
        dash = "-" if len(k) == 1 else "--"
        val = v[0] if isinstance(v, list) else v
        if " " in val:
            val = f'"{val}"'
        args.append(f"{dash}{k} {val}")
    args = " ".join(args)
    return args


@app.route("/spin-up-bot", methods=["POST"])
def spin_up_bot():
    timeout_minutes = int(os.getenv("ROOM_EXPIRE_MINUTES"))
    exp = time.time() + timeout_minutes * 60
    res = requests.post(
        f"https://api.daily.co/v1/rooms",
        headers={"Authorization": f"Bearer {daily_api_key}"},
        json={
            "properties": {
                "exp": exp,
                "start_video_off": True,
                "enable_chat": True,
                "enable_emoji_reactions": False,
                "eject_at_room_exp": True,
                "enable_prejoin_ui": False,
                "enable_screenshare": False,
                "max_participants": 5,
            }
        },
    )
    if res.status_code != 200:
        return (
            jsonify(
                {
                    "error": "Unable to create room",
                    "status_code": res.status_code,
                    "text": res.text,
                }
            ),
            500,
        )
    room_url = res.json()["url"]

    qs = request.query_string.decode()
    qd = parse_qs(qs)
    qd.update({"m": room_url})
    args = convert_dict_to_args(qd)
    cmd = f"/usr/bin/timeout {timeout_minutes}m {sys.executable} vimh.py {args}"
    # cmd = f"{sys.executable} echo-bot.py {args}"
    print(f"server: {cmd}")
    proc = subprocess.Popen([cmd], shell=True)
    subprocesses.append((room_url, proc))

    return jsonify({"room_url": room_url}), 200


AutoIndex(app, browse_root=os.path.curdir)
