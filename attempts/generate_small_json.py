from pathlib import Path

import json
import orjson
from tqdm import tqdm


def read_file(data_dir):
    genes = []
    with tqdm(total=Path(data_dir).stat().st_size, desc="Reading File") as pbar:
        with open(data_dir, "rb") as f:
            for line in f:
                line = orjson.loads(line)
                pbar.update(f.tell() - pbar.n)
                genes.append(line)
            f.close()
    return genes

original_ds3 = read_file("./data/dataset3.json")

# Save first 50 lines to new file
new_ds3 = original_ds3[:50]
with open("./data/sample.json", "wb") as f:
    for i in new_ds3:
        f.write(json.dumps(i).encode())
        f.write("\n".encode())
    f.close()

# Check new file can be opened and read
new_ds3 = read_file("./data/sample.json")
