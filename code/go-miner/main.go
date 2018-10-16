package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"gopkg.in/src-d/go-git.v4"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
	"gopkg.in/src-d/go-git.v4/storage/memory"
)

func main() {
	// Clones the given repository, creating the remote, the local branches
	// and fetching the objects, everything in memory:

	r, err := git.Clone(memory.NewStorage(), nil, &git.CloneOptions{
		URL: "https://github.com/vovak/ima",
	})
	CheckIfError(err)

	// ... retrieves the branch pointed by HEAD
	ref, err := r.Head()
	CheckIfError(err)

	// ... retrieves the commit history
	cIter, err := r.Log(&git.LogOptions{From: ref.Hash()})
	CheckIfError(err)

	// ... just iterates over the commits, printing it
	err = cIter.ForEach(func(c *object.Commit) error {
		//fmt.Println(c)
		PrintCommitDiff(r, c)

		return nil
	})
	CheckIfError(err)
}

func PrintCommitDiff(r *git.Repository, commit *object.Commit) {
	fmt.Println("commit " + commit.Message)

	// omit merge and root commits
	if commit.NumParents() == 1 {
		// get commit's parent
		parent, err := commit.Parent(0)

		// retrieve repository content trees from current and parent revisions
		parentTree, err := parent.Tree()
		currentTree, err := commit.Tree()

		// build a diff between the trees
		changes, err := currentTree.Diff(parentTree)

		// iterate over changed files in the diff
		for _, c := range changes {
			fmt.Println(c)
			patch, err := c.Patch()
			CheckIfError(err)

			filePatches := patch.FilePatches()

			for _, fp := range filePatches {
				if fp.IsBinary() {
					continue
				}
				prev, cur := fp.Files()

				if cur != nil {
					curBlob, err := object.GetBlob(r.Storer, cur.Hash())

					reader, err := curBlob.Reader()
					CheckIfError(err)

					b, err := ioutil.ReadAll(reader)

					fmt.Println(string(b))
				}

				if prev != nil {
					prevBlob, err := object.GetBlob(r.Storer, prev.Hash())
					fmt.Println(prevBlob)
					CheckIfError(err)
				}

				CheckIfError(err)
			}

		}

		CheckIfError(err)
	}
}

// helper methods below are gracefully borrowed from go-git examples
// https://github.com/src-d/go-git/blob/master/_examples/common.go

func CheckArgs(arg ...string) {
	if len(os.Args) < len(arg)+1 {
		Warning("Usage: %s %s", os.Args[0], strings.Join(arg, " "))
		os.Exit(1)
	}
}

// CheckIfError should be used to naively panics if an error is not nil.
func CheckIfError(err error) {
	if err == nil {
		return
	}

	fmt.Printf("\x1b[31;1m%s\x1b[0m\n", fmt.Sprintf("error: %s", err))
	os.Exit(1)
}

// Info should be used to describe the example commands that are about to run.
func Info(format string, args ...interface{}) {
	fmt.Printf("\x1b[34;1m%s\x1b[0m\n", fmt.Sprintf(format, args...))
}

// Warning should be used to display a warning
func Warning(format string, args ...interface{}) {
	fmt.Printf("\x1b[36;1m%s\x1b[0m\n", fmt.Sprintf(format, args...))
}
