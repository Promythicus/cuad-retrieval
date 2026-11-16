#!/usr/bin/env python3
"""A local model on CUAD (contract review), three ways: search then read 5 passages, search then
read 10, and read the whole contract. Everything runs on this machine.

    python bench.py index                 embed every contract's passages and the questions (GPU, before the model is up)
    python bench.py run rag5|rag10|new5|new10|full   ask the model (llama-server on :8081); resumes where it stopped
    python bench.py score                 the table

Contracts: the official CUAD test split (data/test.json), a fixed sample (N, seed below) of those
short enough to be read whole, so the three ways answer exactly the same questions.
"""
import json, math, os, random, re, sys, time, urllib.request

H = os.path.dirname(os.path.abspath(__file__))
N, SEED, MAX_CHARS, CHUNK = int(os.environ.get("BENCH_N", 10)), 7, 120_000, 1200
LLM = os.environ.get("BENCH_LLM", "http://127.0.0.1:8081/v1/chat/completions")
EMB = "BAAI/bge-m3"
OUT = os.path.join(H, "out"); os.makedirs(OUT, exist_ok=True)


def contracts():
    d = json.load(open(os.path.join(H, "data", "test.json")))["data"]
    d = [c for c in d if len(c["paragraphs"][0]["context"]) <= MAX_CHARS]
    random.Random(SEED).shuffle(d)
    return d[:N]


def chunks(text):
    """Passages of whole paragraphs, up to CHUNK characters, with their place in the contract."""
    out, start, buf = [], 0, ""
    pos = 0
    for para in re.split(r"(\n+)", text):
        if len(buf) + len(para) > CHUNK and buf.strip():
            out.append((start, start + len(buf)))
            start, buf = pos, ""
        while len(para) > CHUNK:                      # one very long paragraph
            if buf.strip():
                out.append((start, start + len(buf))); start, buf = pos, ""
            out.append((pos, pos + CHUNK)); pos += CHUNK; para = para[CHUNK:]; start = pos
        buf += para; pos += len(para)
    if buf.strip():
        out.append((start, start + len(buf)))
    return out


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def index():
    import numpy as np, torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB); model = AutoModel.from_pretrained(EMB, torch_dtype=torch.float16).cuda().eval()

    def embed(texts):
        vs = []
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.no_grad():
                v = model(**b).last_hidden_state[:, 0]
            vs.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.concatenate(vs)

    cs = contracts(); qs = sorted({q["question"] for c in cs for q in c["paragraphs"][0]["qas"]})
    np.save(os.path.join(OUT, "q.npy"), embed(qs)); json.dump(qs, open(os.path.join(OUT, "q.json"), "w"))
    for k, c in enumerate(cs):
        text = c["paragraphs"][0]["context"]; ch = chunks(text)
        np.save(os.path.join(OUT, f"c{k}.npy"), embed([text[a:b] for a, b in ch]))
        print(k, c["title"][:60], len(text), "chars", len(ch), "passages", flush=True)


def retrieve(k, c, question, top):
    import numpy as np
    text = c["paragraphs"][0]["context"]; ch = chunks(text)
    qs = json.load(open(os.path.join(OUT, "q.json"))); qv = np.load(os.path.join(OUT, "q.npy"))[qs.index(question)]
    dense = np.argsort(-(np.load(os.path.join(OUT, f"c{k}.npy")) @ qv))
    docs = [toks(text[a:b]) for a, b in ch]; avg = sum(map(len, docs)) / len(docs); q = toks(re.sub(r"Highlight the parts \(if any\) of this contract related to|that should be reviewed by a lawyer\. Details:", " ", question))
    df = {w: sum(w in set(d) for d in docs) for w in set(q)}
    bm = [sum(math.log(1 + (len(docs) - df[w] + .5) / (df[w] + .5)) * d.count(w) * 2.5 / (d.count(w) + 1.5 * (.25 + .75 * len(d) / avg)) for w in set(q)) for d in docs]
    sparse = sorted(range(len(docs)), key=lambda i: -bm[i])
    score = {}
    for rank in (dense.tolist(), sparse):             # reciprocal rank fusion of meaning and keywords
        for r, i in enumerate(rank):
            score[i] = score.get(i, 0) + 1 / (60 + r)
    best = sorted(score, key=lambda i: -score[i])[:top]
    return [ch[i] for i in sorted(best)]


SCHEMA = {"type": "object", "properties": {"found": {"type": "boolean"}, "quotes": {"type": "array", "items": {"type": "string"}, "maxItems": 6}}, "required": ["found", "quotes"]}
RULES = ("You review a contract for a lawyer. Use only the text given. If the contract has language on the point asked about, set found to true and "
         "copy each relevant passage into quotes, word for word, exactly as written. If it has none, set found to false and leave quotes empty. Do not explain.")


def ask(doc, question):
    body = {"model": "local", "temperature": 0, "max_tokens": 700, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": SCHEMA}},
            "messages": [{"role": "system", "content": RULES}, {"role": "user", "content": doc + "\n\nQUESTION: " + question}]}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(LLM, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    try:
        a = json.loads(r["choices"][0]["message"]["content"])
    except Exception:
        a = {"found": False, "quotes": [], "bad": True}
    a["seconds"] = round(time.time() - t, 2); a["prompt_tokens"] = r.get("usage", {}).get("prompt_tokens", 0)
    return a


_best = {}
def BEST():
    if not _best:
        _best.update(json.load(open(os.path.join(OUT, "best.json"))))
    return _best


def run(mode):
    path = os.path.join(OUT, mode + ".jsonl")
    done = {(json.loads(l)["k"], json.loads(l)["question"]) for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a")
    for k, c in enumerate(contracts()):
        text = c["paragraphs"][0]["context"]
        for q in c["paragraphs"][0]["qas"]:
            if (k, q["question"]) in done:
                continue
            if mode == "full":
                got = [(0, len(text))]; doc = "CONTRACT:\n" + text
            elif mode.startswith("new"):                # the best search from retr.py (best.py): 12 or 24 passages of 500 characters
                got = sorted(map(tuple, BEST()[f"{k}|{q['question']}"][:{"new5": 12, "new10": 24}[mode]]))
                doc = "EXCERPTS FROM THE CONTRACT:\n\n" + "\n\n---\n\n".join(text[a:b] for a, b in got)
            else:
                got = retrieve(k, c, q["question"], int(mode[3:]))
                doc = "EXCERPTS FROM THE CONTRACT:\n\n" + "\n\n---\n\n".join(text[a:b] for a, b in got)
            a = ask(doc, q["question"])
            a.update(k=k, question=q["question"], got=got)
            f.write(json.dumps(a) + "\n"); f.flush()
        print(mode, k, c["title"][:50], flush=True)


def score(modes=("rag5", "rag10", "new5", "new10", "full")):
    norm = lambda s: " ".join(s.split())
    rows = []
    for mode in modes:
        path = os.path.join(OUT, mode + ".jsonl")
        if not os.path.exists(path):
            continue
        res = {(r["k"], r["question"]): r for r in map(json.loads, open(path))}
        tp = fp = fn = tn = hit = ans = mq = nq = mg = ng = grounded = secs = ptok = n = bad = 0
        for k, c in enumerate(contracts()):
            text = c["paragraphs"][0]["context"]; flat = norm(text)
            for q in c["paragraphs"][0]["qas"]:
                r = res.get((k, q["question"]))
                if not r:
                    continue
                n += 1; secs += r["seconds"]; ptok += r["prompt_tokens"]; bad += bool(r.get("bad"))
                gold = [] if q["is_impossible"] else q["answers"]
                quotes = [x for x in r["quotes"] if x.strip()] if r["found"] else []
                if gold:
                    ans += 1
                    hit += any(a < g["answer_start"] + len(g["text"]) and g["answer_start"] < b for g in gold for a, b in r["got"])
                tp += bool(gold and quotes); fn += bool(gold and not quotes); fp += bool(quotes and not gold); tn += bool(not gold and not quotes)
                jac = lambda x, y: len(set(toks(x)) & set(toks(y))) / max(1, len(set(toks(x)) | set(toks(y))))
                nq += len(quotes); mq += sum(any(jac(x, g["text"]) >= .5 or norm(g["text"]) in norm(x) or norm(x) in norm(g["text"]) for g in gold) for x in quotes)
                ng += len(gold); mg += sum(any(jac(x, g["text"]) >= .5 or norm(g["text"]) in norm(x) or norm(x) in norm(g["text"]) for x in quotes) for g in gold)
                grounded += sum(norm(x) in flat for x in quotes)
        p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
        rows.append(dict(mode=mode, questions=n, with_answer=ans, passage_recall=round(hit / max(1, ans), 3), found_precision=round(p, 3), found_recall=round(rc, 3),
                         found_f1=round(2 * p * rc / max(1e-9, p + rc), 3), quote_precision=round(mq / max(1, nq), 3), clause_recall=round(mg / max(1, ng), 3),
                         quotes_verbatim=round(grounded / max(1, nq), 3), sec_per_q=round(secs / max(1, n), 2), prompt_tokens=round(ptok / max(1, n)), bad_json=bad))
    for r in rows:
        print(json.dumps(r))
    json.dump(rows, open(os.path.join(OUT, "scores.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    {"index": index, "score": score}.get(sys.argv[1], lambda: run(sys.argv[2]))()
