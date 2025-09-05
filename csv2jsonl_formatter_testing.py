import pandas as pd
import json
import glob
import os

# === 1. Define where your spectra CSVs are ===
data_paths = {
    "Acrylic": "datatrain/Acrylic/*.csv",
}

data_paths_full = {
    "Acrylic": "datatrain/Acrylic/*.csv",
    "Cellulose": "datatrain/Cellulose/*.csv",
    "ENR": "datatrain/ENR/*.csv",
    "EPDM": "datatrain/EPDM/*.csv",
    "HDPE": "datatrain/HDPE/*.csv",
    "LDPE": "datatrain/LDPE/*.csv",
    "Nylon": "datatrain/Nylon/*.csv",
    "PBAT": "datatrain/PBAT/*.csv",
    "PBS": "datatrain/PBS/*.csv",
    "PC": "datatrain/PC/*.csv",
    "PEEK": "datatrain/PEEK/*.csv",
    "PEI": "datatrain/PEI/*.csv",
    "PET": "datatrain/PET/*.csv",
    "PLA": "datatrain/PLA/*.csv",
    "PMMA": "datatrain/PMMA/*.csv",
    "POM": "datatrain/POM/*.csv",
    "PP": "datatrain/PP/*.csv",
    "PS": "datatrain/PS/*.csv",
    "PTFE": "datatrain/PTFE/*.csv",
    "PU": "datatrain/PU/*.csv",
    "PVA": "datatrain/PVA/*.csv",
    "PVC": "datatrain/PVC/*.csv",
}

num = 1
jsonl_out = "testing_data.jsonl"
messages = "what is the microplastic type of the graph with the following points?"


for label, pattern in data_paths.items():
    for file in glob.glob(pattern):
        # === 2. Read CSV (no header) ===
        df = pd.read_csv(file, header=None, names=["x", "y"])

        # === 3. Extract x,y pairs ===
        xy_pairs = df[["x", "y"]].values.tolist()

        # === 4. Downsample (optional, comment out to keep all points) ===
        step = max(1, len(xy_pairs) // 100)  # keep ~2000 points
        xy_pairs = xy_pairs[::step]

        # === 5. Format prompt+completion ===
        messages = messages + f"This is unknown{num} . {xy_pairs}"

        num += 1
item = [
            {"messages" : [
                {"role": "system", "content": "You are an expert microplastic type classifier(There are Acrylic, Cellulose, ENR, EPDM, HDPE, LDPE, NYLON, PBAT, PBS, PC, PEEK, PEI, PET, PLA, PMMA, POM, PP, PS, PTFE, PU, PVA, PVC) based on the graph points that I send to you. the first column is X-axis and the second column is Y-axis. Return only the unknown label and the microplastic type classname"},
                {"role" : "user", "content" : messages},
                
            ]}
        ]
with open(jsonl_out, "w") as f_out:
    f_out.write(json.dumps(item[0]) + "\n")


print(f"Training data saved to {jsonl_out}")
