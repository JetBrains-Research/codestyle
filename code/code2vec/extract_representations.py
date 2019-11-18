import os
import sys

import numpy as np
import pandas as pd

from argparse import ArgumentParser

from representation_pipeline.model.common import Config
from representation_pipeline.model.model import Model



def process_project(folder, n_buckets, min_samples, config):
    processed_folder = ProcessedFolder(folder)

    change_metadata = pd.read_csv(
        processed_folder.change_metadata_file,
        index_col="id",
        usecols=["id", "authorName", "authorEmail", "commitId", "authorTime", "changeType"]
    )
    change_metadata.sort_values(by="authorTime", inplace=True)
    change_metadata["entity"] = change_metadata.apply(
        lambda row: resolve_entity(row["authorName"], row["authorEmail"]),
        axis=1
    )
    # print(change_metadata.head(10))

    change_counts, filtered_authors, total_count = get_change_counts(
        change_metadata["entity"], processed_folder.file_changes_files, min_samples
    )

    # print(change_counts)
    # print(filtered_authors)
    # print(total_count)

    bucket_size = total_count // n_buckets + 1
    change_to_bucket_index = {}
    cur_changes = 0
    cur_bucket = 0
    for change_id in change_metadata.index:

        if change_metadata["entity"].loc[change_id] not in filtered_authors:
            continue

        cur_changes += change_counts[change_id]
        change_to_bucket_index[change_id] = cur_bucket

        while cur_changes >= bucket_size:
            cur_bucket += 1
            cur_changes -= bucket_size

    author_changes_mapping = author_to_changes(
        change_metadata["entity"], filtered_authors, processed_folder.file_changes_files
    )

    config.CHANGES_PATH = processed_folder.file_changes_files
    config.SAVE_PATH = f"models/{folder.split('/')[-1]}/model"
    config.VECTORIZE_PATH = 'models/representation/'
    model = Model(config)
    if not config.LOAD_PATH:
        packs = []
        for author, changes in author_changes_mapping.items():
            for s in range(0, len(changes), config.PACK_SIZE):
                if s + config.PACK_SIZE <= len(changes):
                    packs.append((author, changes[s : s + config.PACK_SIZE]))
        np.random.shuffle(packs)
        # print(packs)
        model.train(packs)
    model.programmer_representation(config.VECTORIZE_PATH, change_metadata["entity"], filtered_authors)


def extract_paths(projects_file, data_root, n_buckets, min_samples, config):
    projects = open(projects_file, "r").readlines()
    projects = [p.strip() for p in projects]
    project_folders = [os.path.join(data_root, p) for p in projects]
    for project_folder in project_folders:
        process_project(project_folder, n_buckets, min_samples, config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--buckets", type=int, default=10, help="Number of blocks in the project history")
    parser.add_argument("--projects_file", type=str, default="../python-miner/projects.txt")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--min_samples", type=int, default=100)
    # parser.add_argument("--train-size", type=float, default=0.8)

    # COPIED FROM CODE2VEC.PY
    parser.add_argument("-dp", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False, nargs='+')
    parser.add_argument("-tp", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False, nargs='+')
    parser.add_argument("-cp", "--changes", dest="changes_path",
                        help="path to changes file", metavar="FILE", required=False, nargs='+')

    is_training = '--train' in sys.argv or '-tr' in sys.argv
    parser.add_argument("-s", "--save", dest="save_path",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-w2v", "--save_word2v", dest="save_w2v",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-t2v", "--save_target2v", dest="save_t2v",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-v", "--vectorize", dest="vectorize_path",
                        help="path to methods for vectorization", metavar="FILE", required=False, nargs='+')
    parser.add_argument('--save_w2v', dest='save_w2v', required=False,
                        help="save word (token) vectors in word2vec format")
    parser.add_argument('--save_t2v', dest='save_t2v', required=False,
                        help="save target vectors in word2vec format")
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a lower model '
                             'size.')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--eval_test', action='store_true')
    parser.add_argument('--num_classes', type=int, required=True)
    # END COPY

    args = parser.parse_args()

    extract_paths(args.projects_file, args.data_root, args.buckets, args.min_samples, Config.get_default_config(args))
