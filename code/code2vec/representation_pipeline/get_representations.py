import os

import pandas as pd
from argparse import ArgumentParser

from time_split import time_split
from get_trained_model import get_trained_model
from resolve_entities import resolve_entities
from util import ProcessedFolder


def get_representations(processed_folder: ProcessedFolder, pack_size: int, embedding_size: int, min_samples: int,
                        n_time_buckets: int):
    if os.path.exists(processed_folder.vectorization_file(pack_size, min_samples)):
        print("Loading previously computed representations")
        return pd.read_csv(processed_folder.vectorization_file(pack_size, min_samples), index_col=0)

    code2vec_model, filtered_authors = get_trained_model(processed_folder, pack_size, embedding_size, min_samples)
    change_authors = resolve_entities(processed_folder)
    change_to_time_bucket = time_split(processed_folder, n_time_buckets)
    print("Computing representations")
    code2vec_model.programmer_representation(
        processed_folder.vectorization_file(pack_size, min_samples), change_authors,
        change_to_time_bucket, filtered_authors
    )
    print("Representations saved on disk")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--pack_size", type=int, required=True)
    parser.add_argument("--embedding_size", type=int, required=True)
    parser.add_argument("--min_samples", type=int, default=0)
    parser.add_argument("--n_time_buckets", type=int, required=True)
    args = parser.parse_args()
    get_representations(ProcessedFolder(args.data_folder), args.pack_size, args.embedding_size, args.min_samples,
                        args.n_time_buckets)
