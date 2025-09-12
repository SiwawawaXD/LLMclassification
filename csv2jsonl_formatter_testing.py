import pandas as pd
import json
import glob
import os

# === 1. Define where your spectra CSVs are ===
data_paths = {
    "PEI": "datatest/PEI/*.csv",
}

data_paths_full = {
    "Acrylic": "datatest/Acrylic/*.csv",
    "Cellulose": "datatest/Cellulose/*.csv",
    "ENR": "datatest/ENR/*.csv",
    "EPDM": "datatest/EPDM/*.csv",
    "HDPE": "datatest/HDPE/*.csv",
    "LDPE": "datatest/LDPE/*.csv",
    "Nylon": "datatest/Nylon/*.csv",
    "PBAT": "datatest/PBAT/*.csv",
    "PBS": "datatest/PBS/*.csv",
    "PC": "datatest/PC/*.csv",
    "PEEK": "datatest/PEEK/*.csv",
    "PEI": "datatest/PEI/*.csv",
    "PET": "datatest/PET/*.csv",
    "PLA": "datatest/PLA/*.csv",
    "PMMA": "datatest/PMMA/*.csv",
    "POM": "datatest/POM/*.csv",
    "PP": "datatest/PP/*.csv",
    "PS": "datatest/PS/*.csv",
    "PTFE": "datatest/PTFE/*.csv",
    "PU": "datatest/PU/*.csv",
    "PVA": "datatest/PVA/*.csv",
    "PVC": "datatest/PVC/*.csv",
}

num = 1
jsonl_out = "testing_data.jsonl"
#messages = "what is the microplastic type of the graph with the following points?"


for label, pattern in data_paths.items():
    for file in glob.glob(pattern):
        # === 2. Read CSV (no header) ===
        df = pd.read_csv(file, header=None, names=["x", "y"])
        print(file)
        # === 3. Extract x,y pairs ===
        xy_pairs = df[["x", "y"]].values.tolist()

        # === 4. Downsample (optional, comment out to keep all points) ===
        step = max(1, len(xy_pairs) // 10000) 
        xy_pairs = xy_pairs[::2]

        # === 5. Format prompt+completion ===
        messages = messages + f"This is unknown{num} . {xy_pairs}"

        num += 1

item = [
            {"role": "system", "content": "You are an expert microplastic type classifier.The amount of points are more detail(twice of your training data) but the overall shape is still similar to what you know. There are total of 23 microplastic types which are Acrylic, Cellulose, ENR, EPDM, HDPE, LDPE, NYLON, PBAT, PBS, PC, PEEK, PEI, PET, PLA, PMMA, POM, PP, PS, PTFE, PU, PVA, PVC) based on the graph points that I send to you. the first column is X-axis and the second column is Y-axis. Classify the unknown and return only the microplastic type classname"},
            {"role" : "user", "content" : messages}
        ]

with open(jsonl_out, "w") as f_out:
    f_out.write(json.dumps(item) + "\n")


print(f"Training data saved to {jsonl_out}")
