import pandas as pd
import json
import glob
import os
import numpy as np
from scipy.interpolate import interp1d

# === 1. Define where your spectra CSVs are ===
data_paths = {
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

jsonl_out = "training_data.jsonl"

with open(jsonl_out, "w") as f_out:
    for label, pattern in data_paths.items():
        for file in glob.glob(pattern):
            # === 2. Read CSV (no header) ===
            df = pd.read_csv(file, header=None, names=["x", "y"])

            # === 3. Extract x,y pairs ===
            xy_pairs = df[["x", "y"]].values.tolist()

            # === 4. Downsample (optional, comment out to keep all points) ===
            step = max(1, len(xy_pairs) // 10000)  # keep ~10000 points
            xy_pairs = xy_pairs[::step]  # can change step variable to exact number ex. change to 2 to make it skip 1 point 

            # === 5. Format prompt+completion ===
            item = [
                {"messages" : [
                    #{"role" : "system"}, ใส่เพิ่มได้
                    {"role" : "user", "content" : f"what is the microplastic type of the graph with the following points? there are  {xy_pairs}"},
                    {"role" : "assistant", "content" : f"{label}"}
                ]}
            ]
            f_out.write(json.dumps(item[0]) + "\n")

print(f"Training data saved to {jsonl_out}")
