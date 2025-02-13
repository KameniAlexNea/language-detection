import os
import datasets
from tqdm import tqdm


ds = datasets.load_dataset("hac541309/open-lid-dataset", split="train")
batch_size = 5_000_000

save_folder = "data/batch"
os.makedirs(save_folder, exist_ok=True)

for i, raws in tqdm(enumerate(ds.iter(batch_size=batch_size))):
    with open(f"{save_folder}/batch{i}-{i+batch_size}.txt", "w") as f:
        f.write("\n".join(raws["text"]))
    with open(f"{save_folder}/label{i}-{i+batch_size}.txt", "w") as f:
        f.write("\n".join(raws["lang"]))
