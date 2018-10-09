""" Find all methods in file with xpath query.
    Extend babelfish Node class with EqualityNode to override `==` operator.
"""
import bblfsh
from EqualityNode import EqualityNode

client = bblfsh.BblfshClient("0.0.0.0:9432")
uast = client.parse("data/MultipleFunctions.java", language="java").uast
uast = EqualityNode(uast)

it = bblfsh.filter(uast, "//MethodDeclaration")
for node in it:
    # print(node)
    for other in bblfsh.filter(uast, "//MethodDeclaration"):
        print(node == other)