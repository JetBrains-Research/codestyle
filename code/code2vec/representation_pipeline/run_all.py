import tensorflow as tf

from argparse import ArgumentParser

from get_representations import get_representations
from merge_aliases_naive import merge_aliases_naive
from util import ProcessedFolder

parser = ArgumentParser()

parser.add_argument("--pack_size", type=int, required=True)
parser.add_argument("--embedding_size", type=int, required=True)
parser.add_argument("--min_samples", type=int, default=0)
parser.add_argument("--n_time_buckets", type=int, required=True)

args = parser.parse_args()

projects = [l.strip() for l in open("../../pythonminer/projects.txt", "r").readlines()]
for p in projects:
    project_folder = ProcessedFolder(f"../../pythonminer/out/{p}/")
    merge_aliases_naive(project_folder)
    tf.reset_default_graph()
    get_representations(project_folder, args.pack_size, args.embedding_size, args.min_samples, args.n_time_buckets)
