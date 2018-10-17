from uast_io import *


def consume_entry(entry):
    blob_id = entry["blob_id"]
    print(blob_id)
    uast = open_uast(blob_id)


def read_trees():
    parse_status = read_parse_status()

    for entry in parse_status:
        if entry["status"] == "OK":
            consume_entry(entry)


read_trees()
