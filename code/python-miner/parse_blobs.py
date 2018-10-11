import os

import bblfsh
import pandas as pd
import pickle

client = bblfsh.BblfshClient("0.0.0.0:9432")

repo_name = "intellij-community"
data_dir = "data/exploded/{}".format(repo_name)
uast_dir = "data/exploded/{}/uast".format(repo_name)

os.makedirs(uast_dir, exist_ok=True)


def is_valid_blob_id(blob_id):
    if not isinstance(blob_id, str):
        return False
    return len(blob_id) == 40


def extract_blob_ids():
    df = pd.read_csv("{}/infos.csv".format(data_dir), index_col=None, na_values='')
    entries = df.to_dict('records')
    print(df.info())
    blobs = []

    def consume_blob_id(blob_id):
        if not is_valid_blob_id(blob_id):
            return
        filename = "{}/blobs/{}".format(data_dir, blob_id)
        if not os.path.exists(filename):
            print("No file {} found for blob {}".format(filename, blob_id))
            return
        blobs.append(blob_id)

    for entry in entries:
        old_blob_id = entry['old_content']
        new_blob_id = entry['new_content']
        consume_blob_id(old_blob_id)
        consume_blob_id(new_blob_id)

    return blobs


def save_uast(blob_id, uast_tree):
    filename = "{}/{}.uast".format(uast_dir, blob_id)
    pickle.dump(uast_tree, open(filename, 'wb'))
    print("Saved UAST to {}".format(filename))

def process_exploded_data():
    blobs_list = extract_blob_ids()
    blob_ids = set(blobs_list)
    print("{} blobs in the dataset, {} unique ".format(len(blobs_list), len(blob_ids)))
    for b in blob_ids:
        blob = load_blob(b)
        blob_uast = parse_blob(b, blob)
        if blob_uast:
            print("blob {} parsed OK, persisting UAST".format(b))
            save_uast(b, blob_uast)
        else:
            print("blob {} failed to parse".format(b))


def load_blob(blob_id):
    filename = "{}/blobs/{}".format(data_dir, blob_id)
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as blob_file:
        blob = blob_file.read()

    return bytes(blob, 'utf-8')


def parse_blob(blob_id, blob):
    if not blob:
        return None
    print("parsing blob {} ({} bytes)".format(blob_id, len(blob)))
    parse_response = client.parse(filename="", contents=blob, language="java")
    if parse_response.status != 0:
        print("Could not parse blob: {}".format(blob_id))
        return None
    return parse_response.uast


process_exploded_data()
