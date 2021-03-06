import subprocess
import sys
import os

projects_list_file = "../pythonminer/git_projects.txt"
projects_dir = "../pythonminer/data/repos/"
short_project_names_file = "../pythonminer/projects.txt"


def extract_name(git_path):
    url = git_path if not git_path.endswith(".git") else git_path[:-4]
    print(url)
    return url.rsplit('/', 1)[-1]


def read_projects():
    projects = []
    with open(projects_list_file, "r") as f:
        for git_path in f:
            projects.append(git_path)
    return projects


def clone_project(project_git_path):
    short_name = extract_name(project_git_path)
    subprocess.run(["git", "clone", project_git_path, projects_dir + short_name])
    return short_name


def write_short_names(short_names):
    with open(short_project_names_file, "w") as f:
        f.writelines("\n".join(short_names))


def run():
    projects = read_projects()
    print("Cloning projects")
    print(projects)
    short_names = []

    for prj in projects:
        sn = clone_project(prj.rstrip())
        short_names.append(sn)

    print("Exploding project repositories")
    print(short_names)

    write_short_names(short_names)

    subprocess.run(["pip", "install", "-r", "../pythonminer/requirements.txt"])
    sys.path.append("../pythonminer")
    from process_repo import RepositoryProcessor

    os.chdir("../pythonminer")

    for sn in short_names:
        repo_processor = RepositoryProcessor(sn)
        repo_processor.explode_repo()

    os.chdir(".")
    subprocess.run(["java", "-jar", "../gumtree-trial/extract-path-contexts.jar"])

    # At this point all necessary data should be in pythonminer/out

    # os.chdir("../code2vec/representation_pipeline")
    # subprocess.run(["python3", "run_all.py", "--pack_size", "4", "--embedding_size", "8", "--min_samples", "4", "--n_time_buckets", "2"])

run()
