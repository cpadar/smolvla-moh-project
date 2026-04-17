"""
Update task description in LeRobot datasets on HuggingFace.
Run once for v2 (128x128) and v3 (512x512).
"""
import os
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import json

NEW_TASK = "Pick up  a red cube, place it next to the green cube, then stack the blue cube on top of the red and green cube to form a pyramid."

def update_dataset_task(repo_id):
    print(f"Updating task for {repo_id}...")
    dataset = LeRobotDataset(repo_id)
    
    # Find the tasks file in local cache
    import glob
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    repo_name = repo_id.replace("/", "--")
    
    # Find tasks parquet file
    pattern = f"{cache_dir}/datasets--{repo_name}/**/tasks.parquet"
    files = glob.glob(pattern, recursive=True)
    print(f"Found task files: {files}")
    
    if not files:
        print(f"No tasks file found for {repo_id}")
        return
    
    import pandas as pd
    for f in files:
        df = pd.read_parquet(f)
        print("Before:", df)
        df['task'] = NEW_TASK
        df.to_parquet(f, index=False)
        print("After:", pd.read_parquet(f))
    
    # Upload updated file to HuggingFace
    api = HfApi()
    for f in files:
        if 'blobs' not in f:  # skip blob cache, only upload snapshots
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo="meta/tasks.parquet",
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"Uploaded updated tasks.parquet to {repo_id}")

update_dataset_task("ceshank01/stack-pyramid-v1-v2")
update_dataset_task("ceshank01/stack-pyramid-v1-v3")
print("Done!")