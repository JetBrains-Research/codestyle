import os
import pickle
import shutil
import unittest

import bblfsh

from EqualityNode import EqualityNode
from tree_differ import TreeDiffer

test_data_dir = "test_data/temp/wrapper"


class TestTreeWrapper(unittest.TestCase):

    def clean_data_dir(self):
        if (os.path.exists(test_data_dir)):
            shutil.rmtree(test_data_dir)

    def setUp(self):
        self.clean_data_dir()
        self.client = bblfsh.BblfshClient("0.0.0.0:9432")
        os.makedirs(test_data_dir, exist_ok=False)

    def wrap_and_restore(self, tree):
        filename = "{}/wrapped_uast".format(test_data_dir)
        with open(filename, 'wb') as f:
            pickle.dump(tree, f)

        with open(filename, 'rb') as f:
            tree_restored = pickle.load(f)

        return tree_restored

    def test_clean_wrapping(self):
        uast = self.client.parse("test_data/wrapper/SingleMethod.java").uast
        tree = EqualityNode(uast)

        tree_restored = self.wrap_and_restore(tree)
        assert tree_restored == tree

    def test_tree_query(self):
        uast = self.client.parse("test_data/wrapper/SingleMethod.java").uast
        tree = EqualityNode(uast)
        methods = TreeDiffer().get_methods(tree)
        assert len(methods) == 1

    def tearDown(self):
        self.clean_data_dir()
