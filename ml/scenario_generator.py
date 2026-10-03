import numpy as np, pandas as pd
def make_scenarios(seed=42):
    r=np.random.default_rng(seed); dates=pd.date_range("2026-01-01",periods=120)
    rows=[]
    for cid in range(1,61):
        base=r.uniform(8,25); cat=r.choice(["RESIDENTIAL","COMMERCIAL","SMALL_INDUSTRY"])
        load=float(r.choice([3,5,10,15])); tr=f"T{r.integers(1,9):02d}"
        for d in dates:
            cons=max(.2,base*(1+.1*np.sin(2*np.pi*d.dayofyear/365))*(.92 if d.dayofweek>=5 and cat=="COMMERCIAL" else 1)+r.normal(0,base*.08))
            v=r.normal(230,2.5); cur=max(.1,cons/24/.92*10+r.normal(0,.3)); p=max(.05,v*cur/1000*.92)
            rows.append([d,f"C{cid:04d}",cons,v,cur,p,"NORMAL","NORMAL",cat,load,tr,1.0])
    df=pd.DataFrame(rows,columns=["timestamp","consumer_id","consumption_kwh","voltage","current","power_kw","meter_status","communication_status","category","sanctioned_load_kw","transformer_id","seasonal_index"])
    cases={"C0001":"THEFT_TAMPERING","C0002":"METER_MALFUNCTION","C0003":"COMMUNICATION_FAILURE","C0004":"LEGITIMATE_VARIATION"}
    out=[]
    for cid,cause in cases.items():
        idx=df.index[df.consumer_id.eq(cid)][-30:]
        if cause=="THEFT_TAMPERING": df.loc[idx,"consumption_kwh"]*=r.uniform(.2,.35,len(idx))
        if cause=="METER_MALFUNCTION": df.loc[idx,"consumption_kwh"]*=r.uniform(.02,.08,len(idx)); df.loc[idx,"meter_status"]="FAULT"
        if cause=="COMMUNICATION_FAILURE":
            mask=r.random(len(idx))<.72; df.loc[idx[mask],"communication_status"]="FAILED"; df.loc[idx[mask],"consumption_kwh"]=np.nan
        if cause=="LEGITIMATE_VARIATION": df.loc[idx,"consumption_kwh"]*=r.uniform(1.45,1.75,len(idx)); df.loc[idx,"seasonal_index"]=1.6
        g=df[df.consumer_id.eq(cid)].copy(); g["label"]=cause; out.append(g)
    n=df[~df.consumer_id.isin(cases)].copy().query("consumer_id in @df.consumer_id.unique()[:12]"); n["label"]="NORMAL"; out.append(n)
    return pd.concat(out,ignore_index=True)
