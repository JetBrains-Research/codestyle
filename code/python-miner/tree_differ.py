import bblfsh

class TreeDiffer:
    def get_method_pairs(self, uast_before, uast_after):
        pass

    def get_methods(self, uast):
        method_nodes = []
        methods_iter = bblfsh.filter(uast, "//MethodDeclaration")
        for m in methods_iter:
            method_nodes.append(m)

        return method_nodes

    def get_name(self, method_node):
        return ""


