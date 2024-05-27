import json
import os

for domain in os.listdir("data"):
    print(f"###### {domain} ######")
    results = json.load(open(fr"results\{domain}.json",'r'))
    for k,v in results.items():
        print("{0: <40}{1}".format(k,round(v["accuracy"],3)))
    print()