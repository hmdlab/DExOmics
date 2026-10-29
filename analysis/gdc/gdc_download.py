import requests, sys, os, pandas as pd, concurrent.futures as cf, time
man=pd.read_csv(sys.argv[1]); outdir=sys.argv[2]; os.makedirs(outdir,exist_ok=True)
def get(row):
    dst=os.path.join(outdir,row.file_id+"__"+row.file_name)
    if os.path.exists(dst) and os.path.getsize(dst)==row.size: return "skip"
    for k in range(4):
        try:
            r=requests.get("https://api.gdc.cancer.gov/data/"+row.file_id,timeout=300,stream=True)
            r.raise_for_status()
            with open(dst+".part","wb") as f:
                for ch in r.iter_content(1<<20): f.write(ch)
            os.replace(dst+".part",dst); return "ok"
        except Exception as e:
            err=str(e); time.sleep(5*(k+1))
    return "FAIL "+err
with cf.ThreadPoolExecutor(6) as ex:
    res=list(ex.map(get,man.itertuples()))
from collections import Counter; print(Counter(r.split()[0] for r in res)); print([r for r in res if r.startswith("FAIL")][:5])
