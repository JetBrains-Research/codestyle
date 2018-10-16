repo_name = "intellij-community"

repo_path = "data/repos/{}".format(repo_name)
data_dir = "data/exploded/{}".format(repo_name)
uast_dir = "{}/uast_zipped".format(data_dir)
blob_dir = "{}/blobs".format(data_dir)

full_repo_data_file = "{}/infos_full.csv".format(data_dir)
incomplete_repo_data_file = "{}/infos_incomplete.csv".format(data_dir)

processed_commits_filename = "{}/processed_commits.csv".format(data_dir)

complete_parse_status_filename = "{}/parse_status.csv".format(data_dir)
incomplete_parse_status_filename = "{}/parse_status_incomplete.csv".format(data_dir)


def get_blob_path(blob_id):
    return "{}/{}".format(blob_dir, blob_id)


def get_uast_path(blob_id):
    return "{}/{}.uast".format(uast_dir, blob_id)
