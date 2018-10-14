import gzip
import os
import pickle
import time

import bblfsh
import pandas as pd

client = bblfsh.BblfshClient("0.0.0.0:9432")

repo_name = "intellij-community"
data_dir = "data/exploded/{}".format(repo_name)
uast_dir = "data/exploded/{}/uast_zipped".format(repo_name)

os.makedirs(uast_dir, exist_ok=True)


def is_valid_blob_id(blob_id):
    if not isinstance(blob_id, str):
        return False
    return len(blob_id) == 40


def extract_blob_ids():
    df = pd.read_csv("{}/infos_full.csv".format(data_dir), index_col=None, na_values='')
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

    tree = uast_tree.__getstate__()
    with gzip.open(filename, 'wb') as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)


def open_uast(blob_id):
    filename = "{}/{}.uast".format(uast_dir, blob_id)
    with gzip.open(filename, 'rb') as f:
        uast_str = pickle.load(f)
        uast = bblfsh.Node()
        uast.__setstate__(uast_str)
    return uast


def save_parse_status(parse_status, complete):
    postfix = "_incomplete" if not complete else ""
    filename = "{}/parse_status{}.csv".format(data_dir, postfix)
    pd.DataFrame.from_records(parse_status).to_csv(filename, index=False)


def read_parse_status():
    filename = "{}/parse_status_incomplete.csv".format(data_dir)
    if not os.path.exists(filename):
        return []
    return pd.read_csv(filename, index_col=None).to_dict('records')


def process_exploded_data():
    blobs_list = extract_blob_ids()
    blob_ids = set(blobs_list)
    blobs_count = len(blob_ids)
    print("{} blobs in the dataset, {} unique ".format(len(blobs_list), blobs_count))

    parse_status = read_parse_status()

    processed = 0
    successful = 0

    for item in parse_status:
        blob_ids.remove(item['blob_id'])
        processed += 1
        if item['status'] == "OK":
            successful += 1

    prev_time = time.time()

    blob_ids = sorted(blob_ids)

    for b in blob_ids:
        #todo remove
        print(b)
        try:
            blob = load_blob(b)
            blob_uast, err = parse_blob(blob)
        except Exception as e:
            blob_uast = None
            err = str(e)

        if blob_uast:
            save_uast(b, blob_uast)
            successful += 1

        parse_status.append({
            "blob_id": b,
            "status": "ERROR" if err else "OK",
            "error": err
        })

        processed += 1

        if processed % 1000 == 0:
            now = time.time()
            print("Processed {} blobs of {}, {} successfully parsed".format(processed, blobs_count, successful))
            print("Last 1000 processed in {} seconds\n".format(now - prev_time))
            prev_time = now
            save_parse_status(parse_status, complete=False)

    save_parse_status(parse_status, complete=True)


def load_blob(blob_id):
    filename = "{}/blobs/{}".format(data_dir, blob_id)
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as blob_file:
        blob = blob_file.read()

    return bytes(blob, 'utf-8')


def parse_blob(blob):
    err = None
    uast = None
    if not blob:
        return None, "Blob not found"
    try:
        parse_response = client.parse(filename="", contents=blob, language="java")
        if parse_response.status != 0:
            err = parse_response.errors
        else:
            uast = parse_response.uast
        return uast, err
    except Exception as e:
        print("Exception on parse attempt: ", e)
        return None, str(e)


process_exploded_data()
