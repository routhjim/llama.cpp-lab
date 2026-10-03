#!/usr/bin/env python3
"""pipe-router.py [--port 8090] [--backends 8080,8081,8082]
One OpenAI-compatible endpoint in front of the LLAMA_SERVER_INSTANCES pipeline (run-qwen38-2x.sh). Each conversation,
keyed by its first two messages (system + first user turn = unique per TB task / chat), is pinned to one instance so
its prompt cache stays warm; a new conversation goes to the instance with the fewest requests in flight.
Streaming (SSE) is passed through chunk by chunk. Other paths go to the first backend."""
import argparse, hashlib, http.client, json, threading
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8090); ap.add_argument("--backends", default="8080,8081,8082")
args = ap.parse_args(); BACK = [int(p) for p in args.backends.split(",")]
lock = threading.Lock(); inflight = {p: 0 for p in BACK}; pins = OrderedDict(); MAXPINS = 4096

def pick(body):
    key = None
    try:
        msgs = json.loads(body).get("messages") or []
        if msgs:
            key = hashlib.sha1(json.dumps(msgs[:2], sort_keys=True).encode()).hexdigest()
    except Exception:
        pass
    with lock:
        if key and key in pins:
            pins.move_to_end(key); p = pins[key]
        else:
            p = min(BACK, key=lambda b: inflight[b])
            if key:
                pins[key] = p
                if len(pins) > MAXPINS: pins.popitem(last=False)
        inflight[p] += 1
    return p

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _proxy(self, method):
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else b""
        chat = method == "POST" and self.path.startswith(("/v1/chat/completions", "/chat/completions", "/v1/completions", "/completion"))
        port = pick(body) if chat else BACK[0]
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=86400)
            hdr = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
            c.request(method, self.path, body=body if body else None, headers=hdr)
            r = c.getresponse()
            self.send_response(r.status)
            chunked = r.getheader("Transfer-Encoding", "").lower() == "chunked" or r.getheader("Content-Length") is None
            for k, v in r.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"): self.send_header(k, v)
            self.send_header("X-Pipe-Instance", str(port))
            if chunked:
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                while True:
                    b = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                    if not b: break
                    self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n"); self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            else:
                data = r.read(); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        finally:
            if chat:
                with lock: inflight[port] -= 1
    def do_GET(self): self._proxy("GET")
    def do_POST(self): self._proxy("POST")

print(f"pipe-router :{args.port} -> {BACK}", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
