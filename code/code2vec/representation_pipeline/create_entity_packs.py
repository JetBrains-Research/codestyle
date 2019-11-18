import os
import pickle
from typing import List, Tuple

import numpy as np
from argparse import ArgumentParser

from compute_occurrences import compute_occurrences
from util import ProcessedFolder


def create_entity_packs(processed_folder: ProcessedFolder, pack_size: int) -> List[Tuple[int, List]]:
    if os.path.exists(processed_folder.entity_packs(pack_size)):
        print("Loading packs for each entity")
        return pickle.load(open(processed_folder.entity_packs(pack_size), 'rb'))

    print("Creating packs for each entity")
    _, _, author_to_changes, _ = compute_occurrences(processed_folder)
    packs = []
    for author, changes in author_to_changes.items():
        np.random.shuffle(changes)
        while len(changes) % pack_size != 0:
            changes.append(np.random.choice(changes))

        for s in range(0, len(changes), pack_size):
            if s + pack_size <= len(changes):
                packs.append((author, changes[s : s + pack_size]))

    pickle.dump(packs, open(processed_folder.entity_packs(pack_size), 'wb'))
    print("Packs saved on disk")

    return packs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--pack_size", type=int, required=True)
    args = parser.parse_args()
    print(create_entity_packs(ProcessedFolder(args.data_folder), args.pack_size))
