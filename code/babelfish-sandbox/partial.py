import bblfsh

client = bblfsh.BblfshClient("0.0.0.0:9432")
uast = client.parse("data/partial_test.py").uast
print(uast)
