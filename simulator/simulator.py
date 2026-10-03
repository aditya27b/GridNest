import random,time,json
def reading(scenario,base=12):
    c=base*random.uniform(.9,1.1); comm="NORMAL"; meter="NORMAL"
    if scenario=="THEFT": c*=random.uniform(.2,.35)
    elif scenario=="METER_FAULT": c*=random.uniform(.02,.08); meter="FAULT"
    elif scenario=="COMMUNICATION_FAILURE" and random.random()<.75: comm="FAILED"; c=None
    elif scenario=="LEGITIMATE": c*=random.uniform(1.45,1.75)
    v=random.uniform(228,232); cur=random.uniform(1,8)
    return {"consumer_id":"C0001","consumption_kwh":c,"voltage":v,"current":cur,"power_kw":v*cur*.92/1000,"meter_status":meter,"communication_status":comm,"scenario":scenario}
if __name__=="__main__":
    ss=["NORMAL","THEFT","METER_FAULT","COMMUNICATION_FAILURE","LEGITIMATE"]
    while True:
        s=random.choice(ss); print(json.dumps(reading(s))); time.sleep(1)
