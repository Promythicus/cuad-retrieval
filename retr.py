#!/usr/bin/env python3
"""Which way of searching a contract finds the clause? Retrieval only, no language model, on all
102 contracts of the CUAD test split (every question that has an answer).

    python retr.py <config> [<config> ...]     run the named configs (see CONFIGS), results in out/retrieval.jsonl
    python retr.py table                       print the table

A config is  embedder:chunk:query:rank[:rerank]  e.g.  bge:1200:full:rrf  or  qwen06:1200:short:rrf:bgerr
  embedder  bge | qwen06 | qwen4 | none      chunk  characters per passage (+t = contract title put before each passage)
  query     full (the CUAD question as written) | short (clause name and its description)
  rank      dense | bm25 | rrf               rerank  bgerr | qwenrr | colbert  (re-orders the first 30)

Scores, at a reading budget of 6,000 and 12,000 characters (5 and 10 passages of 1,200):
  hit   share of questions where at least one passage read holds part of an answer
  span  share of the lawyers' marked clauses that fall in a passage read
"""
import json, math, os, re, sys
import numpy as np
import bench

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "out", "retr"); os.makedirs(OUT, exist_ok=True)
RES = os.path.join(H, "out", "retrieval.jsonl")
MODELS = {"bge": "BAAI/bge-m3", "qwen06": "Qwen/Qwen3-Embedding-0.6B", "qwen4": "Qwen/Qwen3-Embedding-4B"}
TASK = "Instruct: Given a question about a contract, retrieve the passages of the contract that answer it\nQuery: "
BUDGETS, FIRST = (6000, 12000), 30


def data():
    d = json.load(open(os.path.join(H, "data", "test.json")))["data"]
    out = []
    for c in d:
        p = c["paragraphs"][0]
        qs = [(q["question"], [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in q["answers"]]) for q in p["qas"] if not q["is_impossible"] and q["answers"]]
        out.append((c["title"], p["context"], qs))
    return out


def short(q):
    m = re.search(r'related to "(.+?)" that should be reviewed by a lawyer\. Details: (.*)', q, re.S)
    return f"{m.group(1)}: {m.group(2)}" if m else q


def passages(title, text, spec):
    bench.CHUNK = int(spec.rstrip("t"))
    ch = bench.chunks(text)
    head = f"Contract: {title}\n\n" if spec.endswith("t") else ""
    return ch, [head + text[a:b] for a, b in ch]


_loaded = {}


def embedder(name):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if name not in _loaded:
        _loaded.clear(); torch.cuda.empty_cache()
        tok = AutoTokenizer.from_pretrained(MODELS[name], padding_side="left" if name != "bge" else "right")
        _loaded[name] = (tok, AutoModel.from_pretrained(MODELS[name], dtype=torch.float16).cuda().eval())
    tok, model = _loaded[name]

    def embed(texts, tokens=False):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i])); vs = [None] * len(texts)
        step = 32 if name != "qwen4" else 8
        for i in range(0, len(order), step):
            ids = order[i:i + step]
            b = tok([texts[j] for j in ids], padding=True, truncation=True, max_length=1536, return_tensors="pt").to("cuda")
            with torch.no_grad():
                h = model(**b).last_hidden_state
            v = h[:, 0] if name == "bge" else h[:, -1]
            v = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy().astype(np.float16)
            for j, x in zip(ids, v):
                vs[j] = x
        return np.stack(vs)
    return embed


def vectors(name, spec, D):
    """Passage vectors of every contract and the two query forms, kept on disk."""
    path = os.path.join(OUT, f"{name}_{spec}.npz")
    if not os.path.exists(path):
        e = embedder(name); texts = [t for title, text, _ in D for t in passages(title, text, spec)[1]]
        np.savez(path, v=e(texts)); print("embedded", name, spec, len(texts), flush=True)
    qpath = os.path.join(OUT, f"{name}_q.npz")
    qs = sorted({q for _, _, Q in D for q, _ in Q})
    if not os.path.exists(qpath):
        e = embedder(name); pre = TASK if name != "bge" else ""
        np.savez(qpath, full=e([pre + q for q in qs]), short=e([pre + short(q) for q in qs]))
    return np.load(path)["v"].astype(np.float32), np.load(qpath), {q: i for i, q in enumerate(qs)}


def bm25(docs, q):
    avg = sum(map(len, docs)) / len(docs); q = set(q)
    df = {w: sum(w in d for d in map(set, docs)) for w in q}
    return [sum(math.log(1 + (len(docs) - df[w] + .5) / (df[w] + .5)) * d.count(w) * 2.5 / (d.count(w) + 1.5 * (.25 + .75 * len(d) / avg)) for w in q) for d in docs]


def rrf(*ranks):
    s = {}
    for rank in ranks:
        for r, i in enumerate(rank):
            s[i] = s.get(i, 0) + 1 / (60 + r)
    return sorted(s, key=lambda i: -s[i])


def reranker(kind):
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    if kind == "bgerr":
        tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        m = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", dtype=torch.float16).cuda().eval()

        def score(q, docs):
            out = []
            for i in range(0, len(docs), 32):
                b = tok([[q, d] for d in docs[i:i + 32]], padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out += m(**b).logits.view(-1).float().tolist()
            return out
    elif kind == "qwenrr":
        name = "Qwen/Qwen3-Reranker-0.6B"
        tok = AutoTokenizer.from_pretrained(name, padding_side="left"); m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
        yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        pre = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        def score(q, docs):
            out = []
            for i in range(0, len(docs), 16):
                texts = [f"{pre}<Instruct>: Given a question about a contract, retrieve the passages of the contract that answer it\n<Query>: {q}\n<Document>: {d[:5000]}{suf}" for d in docs[i:i + 16]]
                b = tok(texts, padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    lg = m(**b).logits[:, -1].float()
                out += (lg[:, yes] - lg[:, no]).tolist()
            return out
    else:                                              # BGE-M3's many-vector score: each question word finds its best word in the passage
        from huggingface_hub import hf_hub_download
        tok = AutoTokenizer.from_pretrained(MODELS["bge"]); m = AutoModel.from_pretrained(MODELS["bge"], dtype=torch.float16).cuda().eval()
        w = torch.load(hf_hub_download(MODELS["bge"], "colbert_linear.pt"), map_location="cuda")
        W, B = w["weight"].half(), w["bias"].half()

        def many(texts):
            b = tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
            with torch.no_grad():
                h = m(**b).last_hidden_state[:, 1:] @ W.T + B
            h = torch.nn.functional.normalize(h.float(), dim=-1) * b["attention_mask"][:, 1:, None]
            return h, b["attention_mask"][:, 1:]

        def score(q, docs):
            qv, qm = many([q]); out = []
            for i in range(0, len(docs), 16):
                dv, _ = many(docs[i:i + 16])
                s = torch.einsum("qd,ntd->nqt", qv[0], dv).max(-1).values
                out += (s.sum(-1) / qm.sum()).tolist()
            return out
    return score


def run(cfg):
    parts = cfg.split(":"); name, spec, qform, rank = parts[:4]; rr = parts[4] if len(parts) > 4 else None
    D = data(); size = int(spec.rstrip("t"))
    if rank != "bm25":
        V, Q, qi = vectors(name, spec, D)
    scorer = reranker(rr) if rr else None
    if scorer:
        _loaded.clear()
    tot = {b: [0, 0, 0] for b in BUDGETS}; n = spans = at = 0
    for ci, (title, text, qs) in enumerate(D):
        ch, texts = passages(title, text, spec)
        docs = [bench.toks(t) for t in texts]
        for q, gold in qs:
            orders = []
            if rank != "bm25":
                orders.append(np.argsort(-(V[at:at + len(ch)] @ Q[qform][qi[q]].astype(np.float32))).tolist())
            if rank != "dense":
                s = bm25(docs, bench.toks(short(q))); orders.append(sorted(range(len(docs)), key=lambda i: -s[i]))
            order = rrf(*orders) if len(orders) > 1 else orders[0]
            if scorer:
                first = order[:FIRST]; s = scorer(short(q) if qform == "short" else q, [texts[i] for i in first])
                order = [first[i] for i in sorted(range(len(first)), key=lambda i: -s[i])] + order[FIRST:]
            n += 1; spans += len(gold)
            for b in BUDGETS:
                got = [ch[i] for i in order[:max(1, b // size)]]
                held = [any(x < ge and gs < y for x, y in got) for gs, ge in gold]
                tot[b][0] += any(held); tot[b][1] += sum(held); tot[b][2] += sum(y - x for x, y in got)
        at += len(ch)
        if scorer and ci % 10 == 0:
            print(cfg, ci, flush=True)
    r = {"config": cfg, "questions": n, "clauses": spans}
    for b in BUDGETS:
        r[f"hit{b // 1000}k"] = round(tot[b][0] / n, 3); r[f"span{b // 1000}k"] = round(tot[b][1] / spans, 3); r[f"chars{b // 1000}k"] = round(tot[b][2] / n)
    print(json.dumps(r), flush=True)
    open(RES, "a").write(json.dumps(r) + "\n")


def table():
    rows = {}
    for l in open(RES):
        r = json.loads(l); rows[r["config"]] = r
    print(f"{'config':38} {'hit6k':>6} {'span6k':>6} {'hit12k':>6} {'span12k':>7}")
    for r in rows.values():
        print(f"{r['config']:38} {r['hit6k']:6.3f} {r['span6k']:6.3f} {r['hit12k']:6.3f} {r['span12k']:7.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table()
    else:
        done = {json.loads(l)["config"] for l in open(RES)} if os.path.exists(RES) else set()
        for cfg in sys.argv[1:]:
            if cfg not in done:
                run(cfg)
