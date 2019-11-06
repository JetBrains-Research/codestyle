import unittest

import bblfsh
from EqualityNode import EqualityNode
from tree_differ import TreeDiffer


def get_method_id(method_node):
    return [s for s in bblfsh.filter(method_node, "(//MethodInvocation/SimpleName)[1]")][0].token


def assert_pairs_correct(pairs):
    for p in pairs:
        id_before = get_method_id(p['before']['tree'])
        id_after = get_method_id(p['after']['tree'])

        assert id_before == id_after


class TestTreeDiffer(unittest.TestCase):

    def setUp(self):
        self.differ = TreeDiffer()
        self.client = bblfsh.BblfshClient("0.0.0.0:9432")

    def get_tree(self, filename):
        return EqualityNode(self.client.parse(filename).uast)

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

    def test_method_matching_1(self):
        uast_before = self.get_tree("test_data/differ/matching/1/Before.java")
        uast_after = self.get_tree("test_data/differ/matching/1/After.java")

        pairs = self.differ.get_method_pairs(uast_before, uast_after)
        assert_pairs_correct(pairs)

    def test_method_matching_2(self):
        uast_before = self.get_tree("test_data/differ/matching/2/Before.java")
        uast_after = self.get_tree("test_data/differ/matching/2/After.java")

        pairs = self.differ.get_method_pairs(uast_before, uast_after)
        assert_pairs_correct(pairs)

