import requests, json, sys, os, pandas as pd
ROOT = os.path.abspath(os.path.expanduser(os.environ.get("DEXOMICS_DATA", "~/Data/DExOmics")))   # same convention as ../dexo_paths.py
proj=sys.argv[1]  # e.g. TCGA-LIHC
filt={"op":"and","content":[
 {"op":"in","content":{"field":"cases.project.project_id","value":[proj]}},
 {"op":"in","content":{"field":"data_category","value":["Transcriptome Profiling"]}},
 {"op":"in","content":{"field":"data_type","value":["Gene Expression Quantification"]}},
 {"op":"in","content":{"field":"analysis.workflow_type","value":["STAR - Counts"]}},
 {"op":"in","content":{"field":"access","value":["open"]}}]}
fields="file_id,file_name,file_size,cases.samples.sample_type,cases.samples.submitter_id,cases.submitter_id,cases.samples.portions.analytes.aliquots.submitter_id"
r=requests.get("https://api.gdc.cancer.gov/files",params={"filters":json.dumps(filt),"fields":fields,"format":"JSON","size":5000},timeout=120)
hits=r.json()["data"]["hits"]; rows=[]
for h in hits:
    s=h["cases"][0]["samples"][0]; al=s["portions"][0]["analytes"][0]["aliquots"][0]["submitter_id"]
    rows.append(dict(file_id=h["file_id"],file_name=h["file_name"],size=h["file_size"],sample_type=s["sample_type"],sample=s["submitter_id"],case=h["cases"][0]["submitter_id"],aliquot=al))
df=pd.DataFrame(rows); os.makedirs(f"{ROOT}/gdc/{proj.split('-')[1]}", exist_ok=True); df.to_csv(f"{ROOT}/gdc/{proj.split('-')[1]}/manifest_star_counts.csv",index=False)
print(df.sample_type.value_counts()); print("total MB", df["size"].sum()/1e6)
