"""Exercise a running single-slot Inkling text and vision service."""
import argparse
import base64
import json
import struct
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zlib


def solid_png(color):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", 128, 128, 8, 2, 0, 0, 0)
    pixels = (b"\x00" + bytes(color) * 128) * 128
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18060)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    url = base + "/v1/chat/completions"

    def ask(content, model="Inkling-2-Mini"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 128,
        }
        request = Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urlopen(request, timeout=45) as response:
            answer = json.load(response)["choices"][0]["message"]["content"]
        return answer.strip().lower().rstrip(".")

    text = "What is 2 + 2? Reply with just the number."
    assert ask(text) == "4"
    try:
        ask(text, model="invalid-test-model")
    except HTTPError as error:
        assert 400 <= error.code < 500, error.code
    else:
        raise AssertionError("An unknown model was accepted")
    assert ask(text) == "4"

    for name, color in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
        answer = ask([
            {"type": "image_url", "image_url": {"url": solid_png(color)}},
            {"type": "text", "text": "Is the image red or blue? Answer with one word."},
        ])
        assert answer == name, (name, answer)
        print(f"{name} image: {answer}")
    def cancellations():
        with urlopen(base + "/v1/telemetry", timeout=5) as response:
            return json.load(response)["work"]["client_disconnect_cancels"]

    before = cancellations()
    long_request = Request(url, json.dumps({
        "model": "Inkling-2-Mini",
        "messages": [{"role": "user", "content": "List the integers from 1 to 1000, one per line."}],
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0,
    }).encode(), {"Content-Type": "application/json"})
    with urlopen(long_request, timeout=45) as response:
        assert any(line.startswith(b"data:") for line in response)

    deadline = time.monotonic() + 10
    while cancellations() <= before:
        assert time.monotonic() < deadline, "The server did not observe the stream cancellation"
        time.sleep(0.1)
    assert ask(text) == "4"
    print("Text, both vision fixtures, request rejection, cancellation and recovery passed")


if __name__ == "__main__":
    main()
