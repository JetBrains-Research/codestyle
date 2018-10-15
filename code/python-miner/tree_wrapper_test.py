import unittest
import os
import shutil
import pickle

import bblfsh

from EqualityNode import EqualityNode

test_data_dir = "test_data/temp/wrapper"
class TestTreeWrapper(unittest.TestCase):

    def setUp(self):
        self.client = bblfsh.BblfshClient("0.0.0.0:9432")
        os.makedirs(test_data_dir, exist_ok=False)

    def test_clean_wrapping(self):
        uast = self.client.parse("test_data/wrapper/SingleMethod.java").uast
        filename = "{}/single_method_pickle".format(test_data_dir)
        tree = EqualityNode(uast)
        with open(filename, 'wb') as f:
            pickle.dump(tree, f)

        with open(filename,'rb') as f:
            tree_restored = pickle.load(f)

        assert tree_restored == tree

    def tearDown(self):
        shutil.rmtree(test_data_dir)
