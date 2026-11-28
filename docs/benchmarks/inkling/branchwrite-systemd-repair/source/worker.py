import argparse
import json
import socketserver
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
config = json.loads(Path(__file__).with_name("service.json").read_text())
if config["mode"] not in ("serve", "drain"):
    raise ValueError("unsupported mode; allowed modes are serve and drain")
if not args.check:
    class Health(socketserver.StreamRequestHandler):
        def handle(self):
            self.wfile.write((config["mode"] + "\n").encode())

    with socketserver.TCPServer(("127.0.0.1", 0), Health) as server:
        print("health_port", server.server_address[1], flush=True)
        server.serve_forever()
