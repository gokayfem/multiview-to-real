"""Every image-model call goes through `edit`: cache, budget, one attempt, one log line in gen/calls.jsonl.

  - budget: case.json budget.max_calls (or --max-calls) counts every started call of the case's generator
    (successful or failed; cache hits are free). A call that would exceed it is refused BEFORE it starts.
  - no automatic retries: a failure is logged and raised; the stage decides what to do.
  - log: endpoint, sanitised arguments (no key, no data URIs), input sha256 + sizes, request id, the size
    requested and the size READ FROM THE DOWNLOADED FILE (responses have reported null dimensions), elapsed,
    cost note ("unknown" unless the schema states a price; never "free").
  - downloads: https only, size-limited, decoded as an image before being kept.
  - FAL_KEY is read by fal_client from the environment; it is never printed, logged or written.
  - generator "mock" (mockgen.py) writes its output to exactly the same place and logs the same record.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

from . import common
from . import model as model_mod

MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
DOWNLOAD_HOSTS = ("fal.media", "fal.run", "storage.googleapis.com")
DOWNLOAD_DEADLINE_S = 600
DEFAULT_MAX_CALLS = 30


class BudgetError(common.StageError):
    pass


def have_key() -> bool:
    return bool(os.environ.get("FAL_KEY") or os.environ.get("FAL_KEY_ID"))


def log_path(case) -> Path:
    return common.case_dir(case) / "gen" / "calls.jsonl"


def read_log(case) -> list[dict]:
    p = log_path(case)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def calls_started(records: list[dict], generator: str) -> int:
    return sum(1 for r in records if r.get("generator") == generator and r.get("status") in ("success", "error",
                                                                                             "download_error"))


def max_calls(cfg: dict, override: int | None = None) -> int:
    v = override if override is not None else (cfg.get("budget") or {}).get("max_calls", DEFAULT_MAX_CALLS)
    if not isinstance(v, int) or v < 0:
        raise common.StageError(f"budget.max_calls must be an int >= 0, got {v!r}")
    return v


def check_budget(case, cfg: dict, generator: str, planned: int = 1, override: int | None = None) -> dict:
    """Raise BudgetError if `planned` more calls would exceed the budget; else return the budget state."""
    used = calls_started(read_log(case), generator)
    limit = max_calls(cfg, override if override is not None else _env_override())
    if used + planned > limit:
        raise BudgetError(f"budget: {used} of {limit} calls used; {planned} more would exceed it. "
                          f"Raise case.json budget.max_calls (or --max-calls) if intended.")
    return {"used": used, "limit": limit}


def _env_override() -> int | None:
    v = os.environ.get("MVB2R_MAX_CALLS")
    return int(v) if v and v.isdigit() else None


def scrub(text: str) -> str:
    for var in ("FAL_KEY", "FAL_KEY_ID", "FAL_KEY_SECRET"):
        key = os.environ.get(var)
        if key and text:
            text = text.replace(key, "<redacted>")
    return text


def sanitize_arguments(arguments: dict) -> dict:
    clean = {}
    for k, v in arguments.items():
        if any(s in k.lower() for s in ("key", "token", "auth", "secret")):
            continue
        if isinstance(v, list):
            v = [str(u)[:40] + "...(data uri)" if str(u).startswith("data:") else str(u)[:200] for u in v]
        clean[k] = v
    return clean


def cache_key(endpoint: str, prompt: str, size, input_hashes: list[str], extra: dict | None = None) -> str:
    blob = json.dumps({"endpoint": endpoint, "prompt": prompt, "size": list(size) if size else None,
                       "inputs": list(input_hashes), "extra": extra or {}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def append_log(case, record: dict) -> None:
    p = log_path(case)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(scrub(json.dumps(record, sort_keys=True, default=str)) + "\n")


def describe_inputs(case, paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        w, h = common.image_size(p)
        out.append({"path": common.rel(case, p), "sha256": common.sha256_file(p), "width": w, "height": h})
    return out


def find_cached(case, key: str) -> dict | None:
    hit = None
    root = common.case_dir(case)
    for r in read_log(case):
        if r.get("cache_key") != key or r.get("status") != "success":
            continue
        outs = r.get("outputs") or []
        if outs and all((root / o["path"]).is_file() and common.sha256_file(root / o["path"]) == o["sha256"] for o in outs):
            hit = r
    return hit


# ---------------------------------------------------------------- the call
def edit(case, job: dict) -> dict:
    """Run one edit job. job: tag, family, prompt, refs (paths), size (w, h), out (path), views, attempt.
    Returns the log record (outputs[0] has path, sha256, width, height read from the file)."""
    case = common.case_dir(case)
    cfg = common.load_case(case)
    generator = model_mod.generator_of(cfg, job.get("generator"))
    mdl = model_mod.load_model(case, cfg, generator)
    refs = [common.inside_case(case, p) for p in job["refs"]]
    common.inside_case(case, job["out"])
    inputs = describe_inputs(case, refs)
    args = model_mod.build_arguments(mdl, job["prompt"], job["size"], [i["path"] for i in inputs])
    key = cache_key(mdl["endpoint"], job["prompt"], job["size"], [i["sha256"] for i in inputs],
                    {"family": job.get("family"), "views": job.get("views"), "attempt": job.get("attempt", 1)})
    out = Path(job["out"])
    base = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "generator": generator, "endpoint": mdl["endpoint"],
            "tag": job["tag"], "family": job.get("family"), "views": job.get("views"), "cache_key": key,
            "inputs": inputs, "requested_size": list(job["size"]), "cost_note": mdl["price"]["note"]}
    if not job.get("force"):
        hit = find_cached(case, key)
        if hit:
            return _from_cache(case, base, hit, out)
    if generator == "fal":
        _fal_ready()
    check_budget(case, cfg, generator, 1, job.get("max_calls"))
    t0 = time.time()
    try:
        if generator == "mock":
            from . import mockgen
            request_id = mockgen.generate(case, job, out)
        else:
            request_id = _fal_call(mdl, args, refs, out)
    except Exception as exc:
        append_log(case, {**base, "arguments": sanitize_arguments(args), "status": "error",
                          "error": scrub(f"{type(exc).__name__}: {exc}")[:600], "elapsed_s": round(time.time() - t0, 2)})
        raise
    w, h = common.image_size(out)
    rec = {**base, "arguments": sanitize_arguments(args), "status": "success", "request_id": request_id,
           "outputs": [{"path": common.rel(case, out), "sha256": common.sha256_file(out), "width": w, "height": h}],
           "returned_size": [w, h], "size_honoured": [w, h] == list(job["size"]),
           "elapsed_s": round(time.time() - t0, 2)}
    append_log(case, rec)
    return rec


def _from_cache(case, base: dict, hit: dict, out: Path) -> dict:
    src = common.case_dir(case) / hit["outputs"][0]["path"]
    if src.resolve() != out.resolve():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
    o = {**hit["outputs"][0], "path": common.rel(case, out)}
    rec = {**base, "status": "cache_hit", "request_id": hit.get("request_id"), "outputs": [o],
           "returned_size": hit.get("returned_size"), "size_honoured": hit.get("size_honoured"),
           "elapsed_s": 0.0, "cost_note": "none (cache hit)"}
    append_log(case, rec)
    return rec


def _fal_ready():
    """A key in the environment, then fal_client importable; checked BEFORE a call is counted or logged."""
    if not have_key():
        raise common.StageError("FAL_KEY is not set: export it (it is never printed or written), or use "
                                "--generator mock / `targets --images`")
    try:
        import fal_client
    except ImportError as e:
        raise common.StageError("fal-client is not installed (pip install fal-client)") from e
    return fal_client


def _fal_call(mdl: dict, args: dict, refs: list[Path], out: Path) -> str | None:
    fal_client = _fal_ready()
    final = dict(args)
    if mdl.get("image_param"):
        final[mdl["image_param"]] = [fal_client.upload_file(str(p)) for p in refs]
    handle = fal_client.submit(mdl["endpoint"], arguments=final)
    response = handle.get()
    images = (response or {}).get("images") or []
    if not images or not isinstance(images[0], dict) or "url" not in images[0]:
        raise common.StageError("the model's response contained no image URL")
    download(images[0]["url"], out, tuple(mdl.get("download_hosts") or DOWNLOAD_HOSTS))
    return getattr(handle, "request_id", None)


def check_download_url(url: str, hosts=DOWNLOAD_HOSTS) -> str:
    """https and a host ending in one of the allowed suffixes (case.json model.download_hosts) (pure)."""
    u = urllib.parse.urlparse(str(url))
    if u.scheme != "https":
        raise common.StageError(f"refusing to download a non-https link: {str(url)[:60]}")
    host = (u.hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in hosts):
        raise common.StageError(f"refusing to download from {host!r}: not in the allowed hosts {list(hosts)}")
    return url


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    hosts = DOWNLOAD_HOSTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_download_url(newurl, self.hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_content_type(ctype: str | None) -> None:
    if not str(ctype or "").lower().startswith("image/"):
        raise common.StageError(f"download Content-Type {str(ctype)[:40]!r} is not an image")


def download(url: str, dest: Path, hosts=DOWNLOAD_HOSTS, deadline_s: float = DOWNLOAD_DEADLINE_S) -> Path:
    """https from allowed hosts only (redirects too), image Content-Type, size cap, overall deadline;
    the bytes are kept unmodified and must decode as an image within the pixel cap."""
    check_download_url(url, hosts)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    handler = type("Redirect", (_SafeRedirect,), {"hosts": tuple(hosts)})()
    opener = urllib.request.build_opener(handler)
    req = urllib.request.Request(url, headers={"User-Agent": "multiview-blockout-to-real"})
    t0, chunks, total = time.time(), [], 0
    try:
        resp = opener.open(req, timeout=60)
    except (urllib.error.URLError, OSError) as e:
        raise common.StageError(f"download failed: {type(e).__name__}: {str(e)[:200]}") from e
    with resp as r:
        check_download_url(r.geturl(), hosts)
        check_content_type(r.headers.get("Content-Type"))
        while True:
            if time.time() - t0 > deadline_s:
                raise common.StageError(f"download exceeded {deadline_s} s")
            block = r.read(1 << 20)
            if not block:
                break
            total += len(block)
            if total > MAX_DOWNLOAD_BYTES:
                raise common.StageError(f"download larger than {MAX_DOWNLOAD_BYTES >> 20} MB refused")
            chunks.append(block)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(b"".join(chunks))
    try:
        common.check_pixels(tmp)
        from PIL import Image
        with Image.open(tmp) as im:
            im.load()                      # full, bounded decode: a truncated or forged file is refused here
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise common.StageError(f"downloaded file is not an acceptable image: {e}") from e
    os.replace(tmp, dest)
    return dest


def summary(case) -> dict:
    """Calls per generator/endpoint, cache hits, failures, size honoured."""
    out: dict = {}
    for r in read_log(case):
        k = f"{r.get('generator')}:{r.get('endpoint')}"
        s = out.setdefault(k, {"started": 0, "cache_hits": 0, "failed": 0, "size_not_honoured": 0,
                               "cost_note": r.get("cost_note")})
        st = r.get("status")
        if st == "cache_hit":
            s["cache_hits"] += 1
        elif st in ("success", "error", "download_error"):
            s["started"] += 1
            s["failed"] += st != "success"
            s["size_not_honoured"] += st == "success" and r.get("size_honoured") is False
    return out
