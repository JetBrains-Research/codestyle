class EqualityNode:
    """ Since babelfish Node doesn't allow extending, this wrapper class is introduced.
    """
    def __init__(self, n):
        self.internal_type = n.internal_type
        self.properties = n.properties
        self.children = list(map(EqualityNode, n.children))
        self.token = n.token
        self.start_position = n.start_position
        self.end_position = n.end_position
        self.roles = n.roles

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
