import numpy as np, pandas as pd
def add_features(df):
    df=df.copy(); df["timestamp"]=pd.to_datetime(df["timestamp"]); df=df.sort_values(["consumer_id","timestamp"])
    df["communication_failed"]=(df.communication_status!="NORMAL").astype(float)
    df["meter_error"]=(df.meter_status!="NORMAL").astype(float)
    g=df.groupby("consumer_id",group_keys=False)
    df["baseline_kwh"]=g.consumption_kwh.transform(lambda s:s.shift(1).rolling(30,min_periods=7).mean())
    df["rolling_mean_kwh"]=g.consumption_kwh.transform(lambda s:s.shift(1).rolling(7,min_periods=3).mean())
    df["rolling_std_kwh"]=g.consumption_kwh.transform(lambda s:s.shift(1).rolling(7,min_periods=3).std())
    med=df.groupby("consumer_id").consumption_kwh.transform("median")
    df["baseline_kwh"]=df.baseline_kwh.fillna(med); df["rolling_mean_kwh"]=df.rolling_mean_kwh.fillna(df.baseline_kwh); df["rolling_std_kwh"]=df.rolling_std_kwh.fillna(0)
    df["baseline_deviation"]=((df.consumption_kwh-df.baseline_kwh)/df.baseline_kwh.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    df["peer_mean_kwh"]=df.groupby(["category","sanctioned_load_kw"]).consumption_kwh.transform("mean")
    df["peer_deviation"]=((df.consumption_kwh-df.peer_mean_kwh)/df.peer_mean_kwh.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    df["missing_ratio"]=g.consumption_kwh.transform(lambda s:s.isna().rolling(14,min_periods=1).mean())
    df["communication_failure_ratio"]=g.communication_failed.transform(lambda s:s.rolling(14,min_periods=1).mean())
    df["meter_error_count"]=g.meter_error.transform(lambda s:s.rolling(14,min_periods=1).sum())
    df["zero_reading_ratio"]=g.consumption_kwh.transform(lambda s:s.fillna(0).eq(0).rolling(14,min_periods=1).mean())
    for c in ["consumption_kwh","voltage","current","power_kw"]:
        df[c]=pd.to_numeric(df[c],errors="coerce"); df[c]=df.groupby("consumer_id")[c].transform(lambda s:s.interpolate(limit_direction="both"))
    from ml.config import FEATURE_COLUMNS
    df[FEATURE_COLUMNS]=df[FEATURE_COLUMNS].replace([np.inf,-np.inf],np.nan).fillna(0)
    return df
