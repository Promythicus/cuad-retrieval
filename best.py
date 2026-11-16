#!/usr/bin/env python3
"""The best search found by retr.py (500-character passages, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B on the
first 30), run for the pilot's contracts and every one of their questions, with timings. Writes out/best.json
for `bench.py run new5|new10`."""
import json, os, time
import numpy as np, torch
import bench, retr

cs = bench.contracts(); bench.CHUNK = 500
e = retr.embedder("qwen06")
t = time.time(); vec = []; n = chars = 0
for c in cs:
    text = c["paragraphs"][0]["context"]; ch = bench.chunks(text)
    vec.append(e([text[a:b] for a, b in ch]).astype(np.float32)); n += len(ch); chars += len(text)
torch.cuda.synchronize(); t_index = time.time() - t
qs = sorted({q["question"] for c in cs for q in c["paragraphs"][0]["qas"]})
qv = dict(zip(qs, e([retr.TASK + retr.short(q) for q in qs]).astype(np.float32)))
retr._loaded.clear(); torch.cuda.empty_cache()
score = retr.reranker("qwenrr")
out = {}; t = time.time(); m = 0
for k, c in enumerate(cs):
    text = c["paragraphs"][0]["context"]; ch = bench.chunks(text)
    for q in c["paragraphs"][0]["qas"]:
        q = q["question"]
        first = np.argsort(-(vec[k] @ qv[q])).tolist()[:30]
        s = score(retr.short(q), [text[a:b] for a, b in (ch[i] for i in first)])
        out[f"{k}|{q}"] = [ch[first[i]] for i in sorted(range(len(first)), key=lambda i: -s[i])]; m += 1
    print(k, flush=True)
t_search = time.time() - t
json.dump(out, open(os.path.join(bench.OUT, "best.json"), "w"))
r = dict(contracts=len(cs), chars=chars, passages=n, index_seconds=round(t_index, 1), chars_per_second=round(chars / t_index), questions=m, search_seconds_per_question=round(t_search / m, 3))
print(json.dumps(r)); json.dump(r, open(os.path.join(bench.OUT, "best_timing.json"), "w"))
