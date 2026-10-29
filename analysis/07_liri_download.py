#!/usr/bin/env python3
"""
07_liri_download.py — ICGC 25K open bucket から LIRI-JP の exp_seq / specimen / sample / donor を全ドナー分取得

背景: 旧 dcc.icgc.org は 2024-06 に退役。Release 28 の open tier データは S3 互換バケット
  https://object.genomeinformatics.org/icgc25k-open/release_28/data/<PROJ>/<DONOR>/<type>/part-NNNNN.gz
に認証不要で置かれている（確認日 2026-09-20。docs.icgc-argo.org/docs/data-access/icgc-25k-data）。
exp_seq 列は ICGC 標準（gene_model=RefSeq symbol, GRCh37, normalized_read_count=FPKM, raw_read_count=整数）。
列名は release_28/headers/ の該当ヘッダを併せて保存する。

出力: ~/Data/DExOmics/icgc/LIRI-JP/<type>/<donor>__part-NNNNN.gz と headers/
実行: venv/bin/python analysis/07_liri_download.py
"""
import os, sys, time, requests, concurrent.futures as cf
from xml.etree import ElementTree as ET

BASE = "https://object.genomeinformatics.org/icgc25k-open"
PREFIX = "release_28/data/LIRI-JP/"
TYPES = {"exp_seq", "specimen", "sample", "donor"}
from dexo_paths import DATA_ROOT, PROJ_ROOT   # data/project roots: --data-root / DEXOMICS_DATA / ~/Data/DExOmics
OUT = f"{DATA_ROOT}/icgc/LIRI-JP"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

def list_keys(prefix):
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token: params["continuation-token"] = token
        r = requests.get(f"{BASE}/", params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        keys += [c.find(f"{NS}Key").text for c in root.findall(f"{NS}Contents")]
        trunc = root.find(f"{NS}IsTruncated")
        if trunc is None or trunc.text != "true": break
        token = root.find(f"{NS}NextContinuationToken").text
    return keys

def get(key):
    parts = key[len(PREFIX):].split("/")   # <donor>/<type>/part-NNNNN.gz
    if len(parts) != 3 or parts[1] not in TYPES: return None
    donor, typ, part = parts
    dst = f"{OUT}/{typ}/{donor}__{part}"
    if os.path.exists(dst) and os.path.getsize(dst) > 0: return "skip"
    for k in range(4):
        try:
            r = requests.get(f"{BASE}/{key}", timeout=300)
            r.raise_for_status()
            with open(dst + ".part", "wb") as f: f.write(r.content)
            os.replace(dst + ".part", dst); return "ok"
        except Exception as e:
            err = str(e); time.sleep(5 * (k + 1))
    return "FAIL " + key + " " + err

keys = list_keys(PREFIX)
targets = [k for k in keys if len(k[len(PREFIX):].split("/")) == 3 and k[len(PREFIX):].split("/")[1] in TYPES]
donors = sorted({k[len(PREFIX):].split("/")[0] for k in targets})
print(f"keys {len(keys)} | target files {len(targets)} | donors {len(donors)}", flush=True)
for t in TYPES: os.makedirs(f"{OUT}/{t}", exist_ok=True)

os.makedirs(f"{OUT}/headers", exist_ok=True)
for h in list_keys("release_28/headers/"):
    name = os.path.basename(h)
    if any(name.startswith(t) for t in TYPES):
        p = f"{OUT}/headers/{name}"
        if not os.path.exists(p):
            open(p, "wb").write(requests.get(f"{BASE}/{h}", timeout=120).content)
print("headers saved:", os.listdir(f"{OUT}/headers"), flush=True)

with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(get, targets))
from collections import Counter
print(Counter((r or "other").split()[0] for r in res))
fails = [r for r in res if r and r.startswith("FAIL")]
print("fails:", fails[:5])
