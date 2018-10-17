class EqualityNode:
    """ Since babelfish Node doesn't allow extending, this wrapper class is introduced.
    """
    def __init__(self, n):
        self.internal_type = n.internal_type

        self.properties = {}
        for k, v in n.properties.items():
            self.properties[k] = v

        self.children = [EqualityNode(c) for c in n.children]

        self.token = n.token
        self.start_position = SimplePosition(n.start_position)
        self.end_position = SimplePosition(n.end_position)
        self.roles = [r for r in n.roles]

    def __eq__(self, other):
        if isinstance(other, EqualityNode):
            return self.internal_type == other.internal_type and \
                   self.properties == other.properties and \
                   self.token == other.token and \
                   self.roles == other.roles and \
                   self.children == other.children
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class SimplePosition:
    def __init__(self, p):
        self.col = p.col
        self.line = p.line
        self.offset = p.offset
