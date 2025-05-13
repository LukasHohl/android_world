class UINode:
    def __init__(self, attributes, bounding_box, root=False):
        # the attributes from XML
        self.attributes = attributes
        # the bounding box of the UINode is a np.ndarray
        self.bounding_box = bounding_box
        # The children of the UINode
        self.children = []
        # Is this the root?
        self.root = root


    def alternative_str(self):
        """This function is used to describe a node in a simpler way.

        :return: A string describing the node.
        :rtype: str"""
        s = ""
        l = []
        for key, value in self.attributes.items():
            l.append(str(key) + "=\""+str(value)+"\"")
        s += ", ".join(l)
        return s


    def unpack(self, lis, attribute_dict={}):
        """Given a node, this gives the list of descendants that have certain attributes and a bounding box with each side length at least 2 pixel. The list may contain the node itself if it has the required attributes. This function must be called on root to return the modified list.

        :param lis: A list of UINode. This list is returned if this function is called on the root node.
        :type lis: list[UINode]
        :param attribute_dict: A dict filled with the required attributes.
        :type attribute_dict: dict
        :return: A list of descendants that have certain attributes is appended to lis.
        :rtype: list[UINode]"""
        b = False
        # no filtering for attributes if the dict is empty
        if len(attribute_dict) == 0:
            b = True
        # This is a filter that accepts all nodes that have any of the required attributes.
        for i in attribute_dict:
            if i in self.attributes.keys() and self.attributes[i] == attribute_dict[i]:
                b = True
                break
        if b:
            lis.append(self)
        for child in self.children:
            child.unpack(lis=lis, attribute_dict=attribute_dict)
        return lis

