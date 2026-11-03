import asyncio, json, sys, time
import aiohttp
D = json.loads(sys.stdin.readline())
TAKES = 2

async def claude(s, model, system, user, extra):
    body = {"model": model, "max_tokens": 2000, "system": system, "messages": [{"role": "user", "content": user}], **extra}
    async with s.post("https://api.anthropic.com/v1/messages", json=body, headers={
            "x-api-key": KEYS["CLAUDE_API_KEY"], "anthropic-version": "2023-06-01"}) as r:
        j = await r.json()
        if r.status != 200: raise RuntimeError(str(j)[:200])
        return "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")

async def gemini(s, model, system, user, extra):
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": user}]}], **extra}
    async with s.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                      json=body, headers={"x-goog-api-key": KEYS["GEMINI_API_KEY"]}) as r:
        j = await r.json()
        if r.status != 200: raise RuntimeError(str(j)[:200])
        return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"] if not p.get("thought"))

async def openai(s, model, system, user, extra):
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **extra}
    async with s.post("https://api.openai.com/v1/chat/completions", json=body, headers={
            "Authorization": "Bearer " + KEYS["OPENAI_API_KEY"]}) as r:
        j = await r.json()
        if r.status != 200: raise RuntimeError(str(j)[:200])
        return j["choices"][0]["message"]["content"] or ""

FN = {"claude": claude, "gemini": gemini, "openai": openai}

async def run_model(s, label, provider, model, extra):
    out = []
    for smp in D["samples"]:
        for take in range(TAKES):
            t = time.monotonic()
            try:
                text = (await FN[provider](s, model, smp["system"], smp["prompt"], extra)).strip()
            except Exception as e:
                text = f"ERROR: {e}"
            out.append({"model": label, "sample": smp["id"], "take": take, "text": text, "ms": int((time.monotonic() - t) * 1000)})
    print(f"done {label}", file=sys.stderr, flush=True)
    return out

async def main():
    async with aiohttp.ClientSession() as s:
        res = await asyncio.gather(*(run_model(s, *m) for m in D["models"]))
    print(json.dumps([r for rs in res for r in rs]))
asyncio.run(main())
