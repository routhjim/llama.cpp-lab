#!/usr/bin/env python3
"""build_corpus.py OUT [URL] -- render Flash-Next's own agent traffic into a corpus for llama-mtp-dump.
Sources (env): MTP_TB_GLOB = harbor/terminus trajectory.json files, MTP_CC_GLOB = Claude Code session .jsonl files
(only sessions whose assistant turns came from a Flash-Next gguf are used).
Each post-08-31 trajectory (after the expert-swap fix) becomes one conversation rendered through the server's
chat template (/apply-template), so tokens match what the model actually ran on. Documents are separated by
the llama-mtp-dump separator line. A held-out split (every 10th trajectory) goes to OUT.heldout."""
import json, glob, os, sys, urllib.request, datetime
out, url = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8081")
SEP = "<|mtp-doc-sep|>"
def render(msgs):
    r = urllib.request.Request(url + "/apply-template", json.dumps({"messages": msgs}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))["prompt"]
files = []
for pat in ("tb21fn*", "tb4fn*"):
    for f in glob.glob(os.environ.get("MTP_TB_GLOB", f"runs/{pat}/**/agent/trajectory.json"), recursive=True):
        if datetime.date.fromtimestamp(os.path.getmtime(f)) >= datetime.date(2026, 8, 31): files.append(f)
files.sort()
n_docs = [0, 0]; chars = [0, 0]
with open(out, "w") as fo, open(out + ".heldout", "w") as fh:
    for k, f in enumerate(files):
        try: steps = json.load(open(f))["steps"]
        except Exception: continue
        msgs = []
        for s in steps:
            if s["source"] in ("user", "system"):
                msgs.append({"role": s["source"], "content": s.get("message") or ""})
            elif s["source"] == "agent":
                # terminus_2 asks for a JSON reply; the trajectory stores it parsed, so rebuild the JSON the
                # model wrote (indent 2, as in the prompt's example) and feed terminal output back as user text
                msg = s.get("message") or ""
                ana, _, plan = msg.partition("\nPlan:")
                ana = ana.removeprefix("Analysis:").strip()
                cmds = [{"keystrokes": (c.get("arguments") or {}).get("keystrokes", ""), "duration": (c.get("arguments") or {}).get("duration", 1.0)}
                        for c in (s.get("tool_calls") or []) if c.get("function_name") == "bash_command"]
                reply = {"analysis": ana, "plan": plan.strip(), "commands": cmds}
                if any(c.get("function_name") == "mark_task_complete" for c in (s.get("tool_calls") or [])): reply["task_complete"] = True
                m = {"role": "assistant", "content": json.dumps(reply, indent=2, ensure_ascii=False)}
                if s.get("reasoning_content"): m["reasoning_content"] = s["reasoning_content"]
                msgs.append(m)
                obs = "\n".join(str(r.get("content", "")) for r in ((s.get("observation") or {}).get("results")) or [])
                if obs: msgs.append({"role": "user", "content": obs[:20000]})
        if not any(m["role"] == "assistant" for m in msgs): continue
        try: text = render(msgs)
        except Exception as e: print("render failed", f, e, file=sys.stderr); continue
        h = 1 if k % 10 == 9 else 0
        (fh if h else fo).write(text.rstrip("\n") + "\n" + SEP + "\n"); n_docs[h] += 1; chars[h] += len(text)
    # Claude Code sessions served by Flash-Next (model field = the FN gguf), native tool calls
    import re
    cc = []
    for f in glob.glob(os.path.expanduser(os.environ.get("MTP_CC_GLOB", "~/.claude/projects/*/*.jsonl"))):
        if datetime.date.fromtimestamp(os.path.getmtime(f)) < datetime.date(2026, 8, 31): continue
        try: rows = [json.loads(l) for l in open(f)]
        except Exception: continue
        if sum(1 for r in rows if r.get("type") == "assistant" and "Flash-Next" in ((r.get("message") or {}).get("model") or "")) < 3: continue
        msgs, last_id = [], None
        for r in rows:
            m = r.get("message") if isinstance(r.get("message"), dict) else None
            if not m or r.get("type") not in ("user", "assistant"): continue
            c = m.get("content")
            blocks = [{"type": "text", "text": c}] if isinstance(c, str) else (c or [])
            if r["type"] == "assistant":
                if m.get("id") == last_id and msgs and msgs[-1]["role"] == "assistant": a = msgs[-1]
                else: a = {"role": "assistant", "content": ""}; msgs.append(a)
                last_id = m.get("id")
                for b in blocks:
                    if b.get("type") == "thinking": a["reasoning_content"] = a.get("reasoning_content", "") + b.get("thinking", "")
                    elif b.get("type") == "text": a["content"] += b.get("text", "")
                    elif b.get("type") == "tool_use":
                        a.setdefault("tool_calls", []).append({"id": b.get("id"), "type": "function",
                            "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input", {}))}})
            else:
                last_id = None
                for b in blocks:
                    if b.get("type") == "tool_result":
                        tc = b.get("content"); tc = tc if isinstance(tc, str) else "\n".join(x.get("text", "") for x in (tc or []) if isinstance(x, dict))
                        msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": tc[:20000]})
                    elif b.get("type") == "text" and not b.get("text", "").lstrip().startswith("<"):
                        msgs.append({"role": "user", "content": b["text"]})
        if any(m["role"] == "assistant" for m in msgs): cc.append((f, msgs))
    for k, (f, msgs) in enumerate(sorted(cc)):
        try: text = render(msgs)
        except Exception as e: print("render failed", f, e, file=sys.stderr); continue
        h = 1 if k % 10 == 9 else 0
        (fh if h else fo).write(text.rstrip("\n") + "\n" + SEP + "\n"); n_docs[h] += 1; chars[h] += len(text)
    print(f"{len(cc)} Claude Code sessions")
print(f"{len(files)} trajectories -> train {n_docs[0]} docs / {chars[0]/1e6:.1f} M chars, heldout {n_docs[1]} / {chars[1]/1e6:.1f} M chars")
