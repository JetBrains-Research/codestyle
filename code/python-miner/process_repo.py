import os

from git import Repo
from pandas import DataFrame

repo_name = "intellij-community"

repo_path = "data/repos/{}".format(repo_name)
blob_dir = "data/exploded/{}/blobs/".format(repo_name)
os.makedirs(blob_dir, exist_ok=True)


def is_valid(entry):
    if entry.change_type == 'T':
        return False
    path = entry.a_path if entry.change_type == 'D' else entry.b_path
    return path.endswith(".java")


def extract_change_info(commit, entry):
    old_blob_id = None
    new_blob_id = None
    old_path = None
    new_path = None

    if not is_valid(entry):
        return None

    if entry.change_type != 'A':
        old_contents = entry.a_blob.data_stream.read()
        old_path = entry.a_path
        old_blob_id = str(entry.a_blob)
        dump_blob(old_blob_id, old_contents)

    if entry.change_type != 'D':
        new_contents = entry.b_blob.data_stream.read()
        new_path = entry.b_path
        new_blob_id = str(entry.b_blob)
        dump_blob(new_blob_id, new_contents)

    info = {'change_type': entry.change_type,
            'old_path': old_path,
            'new_path': new_path,
            'old_content': old_blob_id,
            'new_content': new_blob_id,
            'commit_id': str(commit),
            'author_name': commit.author.name,
            'author_email': commit.author.email,
            'committer_name': commit.committer.name,
            'committer_email': commit.committer.email,
            'author_time': commit.authored_date,
            'committer_time': commit.committed_date}

    return info


def dump_blob(blob_id, contents):
    filename = "{}/{}".format(blob_dir, blob_id)
    if os.path.exists(filename):
        return filename

    with open(filename, 'wb') as output:
        output.write(contents)

    return filename


def get_changes(commit, parent):
    diff_index = parent.diff(commit)

    change_infos = []

    for entry in diff_index:
        info = extract_change_info(commit, entry)
        if info is not None:
            change_infos.append(info)

    return change_infos


def process_commit(commit):
    parents = commit.parents
    if len(parents) != 1:
        return []

    parent = parents[0]
    return get_changes(commit, parent)


def explode_repo(project_name, path):
    repo = Repo(path)

    total_commits = 0
    for _ in repo.iter_commits():
        total_commits += 1
        if total_commits % 10000 == 0:
            print("Counting commits: {}".format(total_commits))

    # TODO increase for full processing
    limit = 1000
    print("{} commits in the repository. Processing {}".format(total_commits, min(limit, total_commits)))

    processed_count = 0
    df = None

    for c in repo.iter_commits():

        change_infos = process_commit(c)

        if df is None and len(change_infos) > 0:
            df = DataFrame.from_records(change_infos)
        elif len(change_infos) > 0:
            df = df.append(DataFrame.from_records(change_infos))

        processed_count += 1
        if processed_count % 1000 == 0:
            print("Processed {} of {} commits\n".format(processed_count, min(limit, total_commits)))
            print(df.info(memory_usage='deep', verbose=False))
        if processed_count >= limit:
            break

    if df is not None:
        df.to_csv("data/exploded/{}/infos.csv".format(project_name), index=False)


explode_repo(repo_name, repo_path)
