import os
import subprocess

def run():
    os.chdir("../code2vec/representation_pipeline")
    subprocess.run(["python3", "run_all.py", "--pack_size", "4", "--embedding_size", "8", "--min_samples", "1000", "--n_time_buckets", "20", "--n_runs", "30", "--mask_tokens"])

run()

