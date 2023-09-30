from models.base_model import BaseModel


# ____________________________________________________________ DTreeNode
class DTreeNode:
    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.thres = threshold
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None


# ____________________________________________________________ DTreeModel
class DTreeModel(BaseModel):
    def __init__(self):
        pass

    def learn(self, states, targets, **kwargs):
