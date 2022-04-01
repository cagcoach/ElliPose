class ParentTree:
    def __init__(self, root):
        self.root = root

class ParentTreeNode:
    def __init__(self, data, parent:"ParentTreeNode"):
        self.parent = parent
        if not parent is None:
            parent.children.append(self)
        self.children = list()

    def getRoot(self):
        if self.parent is None:
            return self
        else:
            return self.parent.getRoot()

    @property
    def isRoot(self):
        return self.parent is None

    def isLeaf(self):
        return len(self.children) == 0

    def makeThisNodeRoot(self):
        raise NotImplementedError

        parentlist = list()
        curr = self
        while curr.parent is not None:
            parentlist.append(curr.parent)
            curr = curr.parent
