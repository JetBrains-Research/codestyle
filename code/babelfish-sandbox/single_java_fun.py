import bblfsh

client = bblfsh.BblfshClient("0.0.0.0:9432")
uast = client.parse("data/SingleFunction.java", language="java").uast
print(uast)
