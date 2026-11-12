import requests, json, sys, pandas as pd, os
ROOT = os.path.abspath(os.path.expanduser(os.environ.get("DEXOMICS_DATA", "~/Data/DExOmics")))   # same convention as ../dexo_paths.py
proj=sys.argv[1]; kind=sys.argv[2]
specs={
 "cnv":[("data_category",["Copy Number Variation"]),("data_type",["Gene Level Copy Number"]),("analysis.workflow_type",["ASCAT3"])],
 "methy":[("data_category",["DNA Methylation"]),("data_type",["Methylation Beta Value"]),("platform",["Illumina Human Methylation 450"])],
 "maf":[("data_category",["Simple Nucleotide Variation"]),("data_type",["Masked Somatic Mutation"])],
}
content=[{"op":"in","content":{"field":"cases.project.project_id","value":[proj]}},{"op":"in","content":{"field":"access","value":["open"]}}]
content+=[{"op":"in","content":{"field":f,"value":v}} for f,v in specs[kind]]
fields="file_id,file_name,file_size,cases.samples.sample_type,cases.samples.submitter_id,cases.submitter_id,analysis.workflow_type,platform"
r=requests.get("https://api.gdc.cancer.gov/files",params={"filters":json.dumps({"op":"and","content":content}),"fields":fields,"format":"JSON","size":5000},timeout=120)
hits=r.json()["data"]["hits"]; rows=[]
for h in hits:
    cs=h.get("cases",[{}])[0]; ss=cs.get("samples",[{}])
    rows.append(dict(file_id=h["file_id"],file_name=h["file_name"],size=h["file_size"],sample_type=";".join(s.get("sample_type","") for s in ss),sample=";".join(s.get("submitter_id","") for s in ss),case=cs.get("submitter_id",""),workflow=h.get("analysis",{}).get("workflow_type","")))
df=pd.DataFrame(rows); os.makedirs(f"{ROOT}/gdc/{proj.split('-')[1]}", exist_ok=True); df.to_csv(f"{ROOT}/gdc/{proj.split('-')[1]}/manifest_{kind}.csv",index=False)
print(kind, len(df), "files", round(df["size"].sum()/1e9,2), "GB"); print(df.sample_type.value_counts().head(4).to_dict()); print(df.workflow.value_counts().head(3).to_dict())
