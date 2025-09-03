import pandas as pd
import json
import glob
import os

# === 1. Define where your spectra CSVs are ===
data_paths = {
    "Acrylic": "data/Acrylic/*.csv",
    "Cellulose": "data/Cellulose/*.csv",
    "ENR": "data/ENR/*.csv",
    "EPDM": "data/EPDM/*.csv",
    "HDPE": "data/HDPE/*.csv",
    "LDPE": "data/LDPE/*.csv",
    "Nylon": "data/Nylon/*.csv",
    "PBAT": "data/PBAT/*.csv",
    "PBS": "data/PBS/*.csv",
    "PC": "data/PC/*.csv",
    "PEEK": "data/PEEK/*.csv",
    "PEI": "data/PEI/*.csv",
    "PET": "data/PET/*.csv",
    "PLA": "data/PLA/*.csv",
    "PMMA": "data/PMMA/*.csv",
    "POM": "data/POM/*.csv",
    "PP": "data/PP/*.csv",
    "PS": "data/PS/*.csv",
    "PTFE": "data/PTFE/*.csv",
    "PU": "data/PU/*.csv",
    "PVA": "data/PVA/*.csv",
    "PVC": "data/PVC/*.csv",
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
            step = max(1, len(xy_pairs) // 100)  # keep ~2000 points
            xy_pairs = xy_pairs[::step]

            # === 5. Format prompt+completion ===
            item = [
                {"messages" : [
                    {"role" : "user", "content" : f"what is the microplastic type of the graph with the following points? {xy_pairs}"},
                    {"role" : "assistant", "content" : f"{label}"}
                ]}
            ]
            f_out.write(json.dumps(item[0]) + "\n")

print(f"Training data saved to {jsonl_out}")
