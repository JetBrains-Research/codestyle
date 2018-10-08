""" Find all methods in file with xpath query.
"""
import bblfsh

client = bblfsh.BblfshClient("0.0.0.0:9432")
uast = client.parse("data/MultipleFunctions.java", language="java").uast
# print(uast)

it = bblfsh.filter(uast, "//MethodDeclaration")
for node in it:
    print(node)
