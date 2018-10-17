from uast_io import *
from EqualityNode import EqualityNode


def read_uast(entry):
    blob_id = entry["blob_id"]
    print(blob_id)
    uast = open_uast(blob_id)
    return uast


def read_trees():
    parse_status = read_parse_status(complete=True)

    for entry in parse_status:
        if entry["status"] == "OK":
            uast = EqualityNode(read_uast(entry))
            print(uast)


read_trees()
