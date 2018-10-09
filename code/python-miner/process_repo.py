from git import Repo


def get_changes(commit, parent):
    diff_index = commit.diff(parent)
    for entry in diff_index:

        # Only look at modified files for now
        # TODO handle other cases
        if entry.change_type != "M":
            continue

        # Keep this hardcoded for now
        # TODO extract filtering method
        if not entry.a_path.endswith(".java"):
            continue

        print("NEW CONTENTS:")
        print(entry.a_blob.data_stream.read().decode('utf-8'))

        print("OLD CONTENTS:")
        print(entry.b_blob.data_stream.read().decode('utf-8'))

        # TODO persist the contents


def process_commit(commit):
    print(commit)
    parents = commit.parents
    if len(parents) != 1:
        return

    parent = parents[0]
    get_changes(commit, parent)


def process(path):
    repo = Repo(path)

    commits = repo.iter_commits()
    count = 0
    for c in commits:
        process_commit(c)
        count += 1
        if count > 1000:
            return


repo_path = "data/repos/intellij-community/"
process(repo_path)
