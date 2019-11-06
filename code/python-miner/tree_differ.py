import bblfsh


class TreeDiffer:
    def get_method_pairs(self, uast_before, uast_after):
        methods_after = [{"name": self.get_name(m), "tree": m} for m in self.get_methods(uast_after)]
        methods_before = [{"name": self.get_name(m), "tree": m} for m in self.get_methods(uast_before)]

        pairs = []


        for mb in methods_before:
            for ma in methods_after:
                if ma["name"] == mb["name"]:
                    pairs.append({"before": mb, "after": ma})

        return pairs

    def get_methods(self, uast):
        method_nodes = []
        methods_iter = bblfsh.filter(uast, "//MethodDeclaration")
        for m in methods_iter:
            method_nodes.append(m)

        return method_nodes

    def get_name(self, method_node):
        name_nodes = []
        # Retrieve the first SimpleName node
        for n in bblfsh.filter(method_node, "(//SimpleName)[1]"):
            name_nodes.append(n)
        if len(name_nodes) == 0:
            return ""
        return name_nodes[0].token
