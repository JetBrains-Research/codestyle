import os
import pickle
from typing import Dict

import pandas as pd
from argparse import ArgumentParser

from compute_occurrences import compute_occurrences
from util import ProcessedFolder


def time_split(processed_folder: ProcessedFolder, n_time_buckets: int) -> Dict:
    if os.path.exists(processed_folder.time_buckets_split(n_time_buckets)):
        print("Loading split into time-separated buckets")
        return pickle.load(open(processed_folder.time_buckets_split(n_time_buckets), 'rb'))

    print("Splitting into time-separated buckets")
    change_metadata = pd.read_csv(
        processed_folder.change_metadata_file,
        index_col="id",
        usecols=["id", "authorTime"],
        squeeze=True
    )
    change_metadata.sort_values(inplace=True)

    _, change_occurrences, author_to_changes, total_count = compute_occurrences(processed_folder)
    bucket_size = total_count // n_time_buckets + 1
    change_to_time_bucket = {}
    cur_changes = 0
    cur_bucket = 0

    for change_id in change_metadata.index:
        cur_changes += change_occurrences[change_id]
        change_to_time_bucket[change_id] = cur_bucket

        while cur_changes >= bucket_size:
            cur_bucket += 1
            cur_changes -= bucket_size

    pickle.dump(change_to_time_bucket, open(processed_folder.time_buckets_split(n_time_buckets), 'wb'))
    print("Buckets saved on disk")
    return change_to_time_bucket


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--n_time_buckets", type=int, required=True)
    args = parser.parse_args()
    print(time_split(ProcessedFolder(args.data_folder), args.n_time_buckets))
