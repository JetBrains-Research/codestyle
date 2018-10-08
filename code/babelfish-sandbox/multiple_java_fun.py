""" Find all methods in file with xpath query.
    Extend babelfish Node class with EqualityNode to override `==` operator.
"""
import bblfsh
from EqualityNode import EqualityNode

def unposition_node(n):
    n.start_position.line = 0
    n.start_position.offset = 0
    n.start_position.col = 0
    n.end_position.line = 0
    n.end_position.offset = 0
    n.end_position.col = 0


def unposition(tree):
    for n in bblfsh.iterator(tree, bblfsh.TreeOrder.PRE_ORDER):
        unposition_node(n)


client = bblfsh.BblfshClient("0.0.0.0:9432")
uast = client.parse("data/MultipleFunctions.java", language="java").uast
uast = EqualityNode(uast)

it = bblfsh.filter(uast, "//MethodDeclaration")
for node in it:
    # print(node)
    for other in bblfsh.filter(uast, "//MethodDeclaration"):
        print(node == other)