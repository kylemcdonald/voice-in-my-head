from dotenv import load_dotenv

load_dotenv()

import threading
import argparse

from daily import Daily, CallClient, EventHandler


class EchoBot(EventHandler):
    def __init__(self):
        self.client = CallClient(self)
        self.client.set_user_name("Echo")
        self.main_loop_start = threading.Event()
        self.shutdown = threading.Event()

    def run(self, meeting_url):
        self.thread = threading.Thread(target=self.main_loop)
        self.thread.start()
        self.client.join(meeting_url, completion=self.on_joined)
        self.thread.join()

    def on_app_message(self, message, sender):
        if "message" not in message:
            return
        self.client.send_app_message(
            {"room": "main-room", "event": "chat-msg", "message": message["message"]}
        )

    def on_joined(self, data, error):
        self.main_loop_start.set()

    def main_loop(self):
        self.main_loop_start.wait()
        while not self.shutdown.is_set():
            self.shutdown.wait(1)

    def end(self):
        self.shutdown.set()
        self.client.leave()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("-m", "--meeting", required=True, help="Meeting URL")
    args, _ = parser.parse_known_args()

    Daily.init()
    app = EchoBot()

    try:
        app.run(args.meeting)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("error while running app, ending app")
        print(e)
    finally:
        app.end()


if __name__ == "__main__":
    main()
