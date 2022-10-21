
from treelib import Tree as TreeTree
import random

class OpInfo():
    # struct describing arithmetic operations
    def __init__(self, func, name):
        self.func = func # func pointer/lambda
        self.name = name # string
class _OperationHolder():
    # fake enum
    def __init__(self):
        self.ADD = OpInfo(lambda a, b: a+b, "ADD")
        self.MULT = OpInfo(lambda a, b: a*b, "MULT")
OPERATIONS = _OperationHolder() # instantiate enum

class Node:

    def __init__(self, operation, op_a, op_b, leaf=False):
        """
        :param operation: OpInfo describing the node's operation
        :param op_a: Pointer to child node
        :param op_a: Pointer to other child node
        """
        self.operation = operation.func # function
        self.name = operation.name # string
        self.op_a = op_a # node ptr
        self.op_b = op_b # node ptr

        self.is_leaf = False

        self.assocs = {}
        if not leaf:
            if self.op_a.name == self.name and not self.op_a.is_leaf:
                self.assocs = self.op_a.assocs.copy()
            else:
                self.assocs[self.op_a.id] = 1
            if self.op_b.name == self.name and not self.op_b.is_leaf:
                for key in self.op_b.assocs:
                    if key in self.assocs.keys():
                        self.assocs[key] += 1
                    else:
                        self.assocs[key] = 1
            else:
                if self.op_b.id in self.assocs.keys():
                    self.assocs[self.op_b.id] += 1
                else:
                    self.assocs[self.op_b.id] = 1

        self.id = random.randrange(2**62)

        # self.prev_arg = None # the previous arg seen by node
        # self.cache = None # cache of last output

    def change_operation(self, new_operation):
        """
        Change the operation that this node executes.
        :param new_operation: OpInfo for the new operation
        """
        self.operation = new_operation.func
        self.name = new_operation.name

    def run(self, arg):
        """
        Recursively execute the circuit and return the output of this node.
        Caches outputs incase a node is the child of multiple parents.
        :param arg: The variable input for the given execution
        """
        return self.operation(
            self.op_a.run(arg), self.op_b.run(arg)
        )

    def show(self, tree=None, parent=None):
        """
        Use treelib to visualize this node and its children in the terminal.
        :param tree: Instantiated tree object
        :param parent: Pointer back to parent TreeNode
        NOTE: parameters are intended for internal use, not user (too lazy to wrap)
        """
        # make sure there is a tree
        my_tree = tree
        if tree == None:
            my_tree = TreeTree()
        # create tree node for this
        my_node = None
        if parent == None:
            my_node = my_tree.create_node(self.name)
        else:
            my_node = my_tree.create_node(self.name, parent=parent)
        # recursively have children add themselves to the tree
        self.op_a.show(tree=my_tree, parent=my_node)
        self.op_b.show(tree=my_tree, parent=my_node)
        # show if this owns the tree
        if tree == None:
            my_tree.show()

    def size(self):
        return 1 + self.op_a.size() + self.op_b.size()

class LeafNode(Node):

    def __init__(self, val=None):
        """
        :param val: None represents an arg input, otherwise is constant of value val. (Default: None)
        """
        Node.__init__(self, OPERATIONS.ADD, None, None, leaf=True)
        # is this arg of const?
        self.arg_based = True
        if val != None:
            self.arg_based = False
        # const or None
        self.val = val
        self.is_leaf = True

    def run(self, arg):
        """
        Returns the value of this node. Reaching here is 'stopping condition' of recursive execution.
        :param arg: Variable input for given execution, used if self.arg_based is True
        """
        if self.arg_based:
            return arg
        return self.val

    def show(self, tree=None, parent=None):
        """
        This should typically be called by parent node. Adds this node to visualization tree.
        :param tree: Instantiated tree object
        :param parent: Pointer back to parent TreeNode
        """
        # make sure there is a tree
        my_tree = tree
        if tree == None:
            my_tree = TreeTree()
        # const else x for input
        my_name = 'x'
        if not self.arg_based:
            my_name = str(self.val)
        # link with tree
        if parent == None:
            my_tree.create_node(my_name)
        else:
            my_tree.create_node(my_name, parent=parent)
        # show if this owns the tree
        if tree == None:
            my_tree.show()

    def size(self):
        return 0

def treequals(a, b, mem=None):
    if mem == None:
        mem = {}
    if (a.id, b.id) in mem.keys():
        return mem[(a.id, b.id)]

    if a.is_leaf != b.is_leaf:
        mem[(a.id, b.id)] = False
        return False
    
    if not a.is_leaf:
        if a.name != b.name:
            mem[(a.id, b.id)] = False
            return False
        mem[(a.id, b.id)] = (
            (treequals(a.op_a, b.op_a, mem) and treequals(a.op_b, b.op_b, mem)) or
            (treequals(a.op_a, b.op_b, mem) and treequals(a.op_b, b.op_a, mem))
        )
        return mem[(a.id, b.id)]

    mem[(a.id, b.id)] = a.val == b.val
    return a.val == b.val