import json
import orjson
from tqdm import tqdm
from pathlib import Path

genes=[]
data_dir = "./data/dataset3.json"
with tqdm(total=Path(data_dir).stat().st_size, desc="Reading File") as pbar:
    with open(data_dir, "rb") as f:
        for line in f:
            line = orjson.loads(line)
            pbar.update(f.tell() - pbar.n)
            genes.append(line)
        f.close()

# Save first 50 lines to new file
genes = genes[:50]
with open("./data/sample.json", "wb") as f:
    for i in genes:
        f.write(json.dumps(i).encode())
        f.write("\n".encode())
    f.close()

# Check new file can be opened and read
a=[]
data_dir = "./data/sample.json"
with tqdm(total=Path(data_dir).stat().st_size, desc="Reading File") as pbar:
    with open(data_dir, "rb") as f:
        for line in f:
            line = orjson.loads(line)
            pbar.update(f.tell() - pbar.n)
            a.append(line)
        f.close()