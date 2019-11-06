import gzip
import os
import pickle

import bblfsh
import pandas as pd

from path_util import *


def save_parse_status(parse_status, complete):
    filename = complete_parse_status_filename if complete else incomplete_parse_status_filename
    pd.DataFrame.from_records(parse_status).to_csv(filename, index=False)


def read_parse_status(complete=False):
    filename = complete_parse_status_filename if complete else incomplete_parse_status_filename
    if not os.path.exists(filename):
        return []
    return pd.read_csv(filename, index_col=None).to_dict('records')


def save_uast(blob_id, uast_tree):
    filename = get_uast_path(blob_id)

    tree = uast_tree.__getstate__()
    with gzip.open(filename, 'wb') as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)


def open_uast(blob_id):
    filename = get_uast_path(blob_id)
    with gzip.open(filename, 'rb') as f:
        uast_str = pickle.load(f)
        uast = bblfsh.Node()
        uast.__setstate__(uast_str)
    return uast
