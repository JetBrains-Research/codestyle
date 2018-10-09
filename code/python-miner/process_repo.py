import bblfsh
import time
from git import Repo

client = bblfsh.BblfshClient("0.0.0.0:9432")


def get_changes(commit, parent):
    diff_index = commit.diff(parent)
    requests = 0
    successful_requests = 0
    for entry in diff_index:

        # Only look at modified files for now
        # TODO handle other cases
        if entry.change_type != "M":
            continue

        # Keep this hardcoded for now
        # TODO extract filtering method
        if not entry.a_path.endswith(".java"):
            continue

        new_contents = entry.a_blob.data_stream.read()
        # print("NEW CONTENTS:")
        # print(new_contents)

        new_parse_response = client.parse(filename="", contents=new_contents, language="java")
        requests += 1
        if new_parse_response.status == 0:
            successful_requests += 1

        old_contents = entry.b_blob.data_stream.read()
        # print("OLD CONTENTS:")
        # print(old_contents)

        old_parse_response = client.parse(filename="", contents=old_contents, language="java")
        requests += 1
        if old_parse_response.status == 0:
            successful_requests += 1
        # print(old_parse_response.status)

        # TODO persist the contents

    return requests, successful_requests


def process_commit(commit):
    print(commit)
    parents = commit.parents
    if len(parents) != 1:
        return

    parent = parents[0]
    return get_changes(commit, parent)


def process(path):
    repo = Repo(path)

    commits = repo.iter_commits()
    count = 0
    limit = 100
    start = time.time()
    total = 0
    success = 0
    for c in commits:
        req, succ = process_commit(c)
        total += req
        success += succ
        count += 1
        print(count)
        if count >= limit:
            break
    end = time.time()

    print("Processed {} commits in {} seconds".format(limit, end - start))
    print("Made {} parse requests, of which {} were successful".format(total, success))


repo_path = "data/repos/intellij-community/"
process(repo_path)
