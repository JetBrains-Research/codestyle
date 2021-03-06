import os
from typing import Tuple, List

from argparse import ArgumentParser

from compute_occurrences import compute_occurrences
from create_entity_packs import create_entity_packs
from util import ProcessedFolder
from model.common import Config
from model.model import Model


def get_trained_model(processed_folder: ProcessedFolder, pack_size: int, embedding_size: int,
                      min_samples: int, n_run: int, total_runs: int, mask_tokens: bool) -> Tuple[Model, List]:

    print("Gathering model configuration")
    author_occurrences, _, _, _ = compute_occurrences(processed_folder)
    filtered_authors = []
    for author, count in author_occurrences.most_common():
        if count >= min_samples:
            filtered_authors.append(author)
    print("{} authors have at least {} samples".format(len(filtered_authors), min_samples))

    n_tokens = processed_folder.n_tokens()
    n_paths = processed_folder.n_paths()
    print("Found {} tokens and {} paths".format(n_tokens, n_paths))

    load_path = os.path.join(processed_folder.trained_model_folder(pack_size, min_samples, mask_tokens), "model")

    config = Config.get_representation_config(dataset_folder=processed_folder.folder, load_path=load_path,
                                              changes_path=processed_folder.file_changes,
                                              n_tokens=n_tokens, n_paths=n_paths,
                                              n_entities=max(filtered_authors),
                                              embedding_size=embedding_size, pack_size=pack_size,
                                              n_run=n_run, total_runs=total_runs)

    code2vec_model = Model(config)
    if config.LOAD_PATH == '':
        print("Did not find a pretrained model")
        packs = create_entity_packs(processed_folder, pack_size)
        packs = [pack for pack in packs if pack[0] in filtered_authors]
        code2vec_model.train(packs, mask_tokens)
        print("Completed training")

    return code2vec_model, filtered_authors


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--pack_size", type=int, required=True)
    parser.add_argument("--embedding_size", type=int, required=True)
    parser.add_argument("--min_samples", type=int, default=0)
    parser.add_argument("--mask_tokens", type=bool, default=False)
    args = parser.parse_args()

    get_trained_model(ProcessedFolder(args.data_folder), args.pack_size, args.embedding_size, args.min_samples,
                      0, 0, args.mask_tokens)
