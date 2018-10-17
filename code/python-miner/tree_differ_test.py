import unittest

import bblfsh

from tree_differ import TreeDiffer


class TestTreeDiffer(unittest.TestCase):

    def setUp(self):
        self.differ = TreeDiffer()
        self.client = bblfsh.BblfshClient("0.0.0.0:9432")

    def get_tree(self, filename):
        return self.client.parse(filename).uast

    def test_single_method_count(self):
        uast = self.get_tree("test_data/differ/SingleMethod.java")
        methods = self.differ.get_methods(uast)
        assert len(methods) == 1

    def test_single_method_name(self):
        uast = self.get_tree("test_data/differ/SingleMethod.java")
        methods = self.differ.get_methods(uast)
        assert self.differ.get_name(methods[0]) == "main"

    def test_single_method_count_inner_class(self):
        uast = self.get_tree("test_data/differ/SingleMethodInnerClass.java")
        methods = self.differ.get_methods(uast)
        assert len(methods) == 1

    def test_single_method_name_inner_class(self):
        uast = self.get_tree("test_data/differ/SingleMethodInnerClass.java")
        methods = self.differ.get_methods(uast)
        assert self.differ.get_name(methods[0]) == "main"
