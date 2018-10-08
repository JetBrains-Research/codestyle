""" Find a method declaration in a file with single class containing single method.
"""
import bblfsh

client = bblfsh.BblfshClient("0.0.0.0:9432")
uast = client.parse("data/SingleFunction.java", language="java").uast
# print(uast)

it = bblfsh.iterator(uast, bblfsh.TreeOrder.PRE_ORDER)
for node in it:
    if node.internal_type == "MethodDeclaration":
        print(node)
