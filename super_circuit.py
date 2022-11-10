
from circuit import OPERATIONS, OpInfo
import numpy as np
from sparse_poly import SparsePoly

class SuperNode:

    def __init__(self, operands=None, operation=None, is_leaf=False, arg=None, val=None):

        # make sure args are coherant
        if not is_leaf and (operands == None or operation == None):
            raise ValueError("A. SuperNode must have operands and operation if NOT leaf.")
        if not is_leaf and (arg != None or val != None):
            raise ValueError("B. SuperNode NOT have arg nor val if NOT leaf.")
        if is_leaf and (operands != None or operation != None):
            raise ValueError("C. SuperNode must NOT have operands and operation if leaf.")
        if is_leaf and (arg == None and val == None):
            raise ValueError("D. SuperNode must have arg or val if leaf.")

        self.outputs = set() # store pointer to nodes that take this as input
        self.depends_on = set() # store the nodes that this depends on

        self.is_leaf = is_leaf # whether this is leaf
        if self.is_leaf:
            # if leaf, then either const val or arg
            self.arg = arg # number representing index of input
            self.val = val

        else:
            # else does operation
            self.operation = operation
            self.operands = operands

            for operand in operands:
                operand.set_output(self)
                self.depends_on.add(operand)
                self.depends_on = self.depends_on | operand.depends_on
    

    def set_output(self, out):
        # set node as output of self
        self.outputs.add(out)

    def remove_output(self, out):
        # remove node as output of self
        self.outputs.discard(out)


    def change_input(self, old_input, new_input):
        if self.is_leaf:
            raise ValueError("Cannot change the input to a leaf!")

        # check to avoid circular dependencies
        if new_input is self or self in new_input.depends_on:
            return False

        old_input.remove_output(self)
        new_input.set_output(self)

        self.operands[self.operands.index(old_input)] = new_input

        self._update_depends()
        if self in self.depends_on:
            raise RuntimeError("Node is dependent on itself")
        return True
        
    def _update_depends(self):

        new_depends = set(self.operands)
        for op in self.operands:
            new_depends = new_depends | op.depends_on

        if new_depends != self.depends_on:
            self.depends_on = new_depends
            for out in self.outputs:
                out._update_depends()

class SuperCircuit:

    def __init__(self, n_args, costs={OPERATIONS.MULT: 1, OPERATIONS.ADD: 1}):
        # number of inputs that this tree takes
        self.n_args = n_args

        # all nodes that are not leaves
        self.nodes = set()

        # all nodes including leaves
        self.all_nodes = set()

        self.leaf_nodes = set()

        self._unclean = set()

        # hold const leaves in dict with val -> node
        self.val_leaves = {}

        # hold arg leaves in list
        self.arg_leaves = [
            SuperNode(is_leaf=True, arg=i) for i in range(n_args)
        ]
        for a in self.arg_leaves:
            self.all_nodes.add(a)
            self.leaf_nodes.add(a)

        # keep track of how mush operations cost
        self.op_costs = costs.copy()
        # keep track of how much this tree costs
        self.cost = 0

        # the action that undoes the last action
        self.undo_stack = []

        # get root node of tree
        self.root = None

    def getValNode(self, val, save=True):
        """
        :return: a leaf node of the given value. Creates one if it doesn't already exist
        """
        if val in self.val_leaves:
            return self.val_leaves[val]
        else:
            new_node = SuperNode(is_leaf=True, val=val)
            if save:
                self.val_leaves[val] = new_node
                self.all_nodes.add(new_node)
                self.leaf_nodes.add(new_node)
            self._unclean.add(new_node)
            return new_node


    def getArgNode(self, arg):
        """
        :return: The arg node of the corresponding index
        """
        return self.arg_leaves[arg]


    def getRoot(self):
        """
        :return: The root node
        """
        return self.root


    def _addNode_wrap(self, args):
        self.addNode(None, None, args[0])
        for i in range(args[1]):
            self.undo()
    def addNode(self, operation, operands, node=None):
        """
        Add a new node to the tree
        :arg operation: the operation that this node performs
        :arg operands: Iterable of operands that this will take as inputs
        """

        # create the new node
        new_node = node
        if node == None:
            new_node = SuperNode(operation=operation, operands=operands)
        else:
            operation = node.operation
            operands = node.operands
        # add it to the register
        self.nodes.add(new_node)
        self.all_nodes.add(new_node)
        self._unclean.add(new_node)
        self.cost += self.op_costs[operation]

        # this is the new root node if it depends on the old root
        old_root = None
        for op in operands:
            if self.root == None or op is self.root:
                old_root = op
                self.root = new_node
                break
        if old_root == None:
            for op in operands:
                old_root = op
                break
        
        self.undo_stack.append({"func": self._removeNode_wrap, "args": (new_node, old_root)})
        return new_node

    def _removeNode_wrap(self, args):
        self.removeNode(args[0], args[1])
    def removeNode(self, node, keeper):
        if not keeper in node.operands:
            raise RuntimeError("Tried removing node, but keeper is not operand")

        num_changes = len(node.outputs)
        coped = node.outputs.copy()
        for out in coped:
            succ = self.change_input(out, node, keeper)
            if not succ:
                raise RuntimeError("fuck, input change failed in removeNode (this shouldn't happen)")

        self.cost -= self.op_costs[node.operation]
        self.nodes.discard(node)
        self.all_nodes.discard(node)


        if node is self.root:
            self.root = keeper

        self.undo_stack.append({"func": self._addNode_wrap, "args": (node, num_changes)})
        
        return True


    def _change_operation_wrap(self, args):
        return self.change_operation(args[0], args[1])
    def change_operation(self, node, operation):
        if node.is_leaf:
            raise ValueError("Cannot change the operation of a leaf node!")
        
        if node.operation == operation:
            return False

        self.cost -= self.op_costs[node.operation]
        self.cost += self.op_costs[operation]
        self.undo_stack.append({"func": self._change_operation_wrap, "args": (node, node.operation)})
        node.operation = operation
        return True


    def _change_input_wrap(self, args):
        return self.change_input(args[0], args[1], args[2])
    def change_input(self, node, old_input, new_input):
        succ = node.change_input(old_input, new_input)

        if succ:
            self.undo_stack.append({"func": self._change_input_wrap, "args": (node, new_input, old_input)})

        return succ

    def undo(self):
        if self.undo_stack == []:
            print("WARNING: undo called for circuit with empty undo stack")
        old_len = len(self.undo_stack)

        undo_action = self.undo_stack.pop()

        succ = undo_action["func"](undo_action["args"])
        if succ is False:
            print("WARNING: undo failed in circuit")
        
        while len(self.undo_stack) > old_len-1:
            if len(self.undo_stack) == 0:
                break
            self.undo_stack.pop()


    def clean(self):
        self.undo_stack = []

        to_remove = set()
        for n in self._unclean:
            if n not in self.root.depends_on and not n is self.root and not n.is_leaf:
                to_remove.add(n)

        for n in to_remove:
            self.nodes.discard(n)
            self.all_nodes.discard(n)
            self._unclean.discard(n)
        
        return len(to_remove)


    def recalc_cost(self):
        self.cost = 0
        for n in self.root.depends_on:
            if not n.is_leaf:
                self.cost += self.op_costs[n.operation]
        if not self.root.is_leaf:
            self.cost += self.op_costs[self.root.operation]


    def evaluate(self, args):

        outs = {}
        track_stack = []
        curr = self.root

        while True:
            if curr in track_stack:
                raise RuntimeError("Circular dependency in evaluate!")

            if curr.is_leaf:
                if curr.arg != None:
                    if curr is self.root:
                        return args[curr.arg]
                    outs[curr] = args[curr.arg]
                    curr = track_stack.pop()
                else:
                    if curr is self.root:
                        return np.full_like(args[0], curr.val)
                    outs[curr] = np.full_like(args[0], curr.val)
                    curr = track_stack.pop()

            else:
                for op in curr.operands:
                    if op not in outs.keys():
                        track_stack.append(curr)
                        curr = op
                        break
                else:
                    if curr not in outs.keys():

                        if len(curr.operands) != 2:
                            raise ValueError("Node does not have two operands! " + str(len(curr.operands)))

                        outs[curr] = curr.operation.func(
                            outs[curr.operands[0]],
                            outs[curr.operands[1]]
                        )
                        if curr is self.root:
                            break

                    curr = track_stack.pop()

        return outs[self.root]

    def get_poly(self):

        outs = {}
        track_stack = []
        curr = self.root

        while True:
            if curr in track_stack:
                raise RuntimeError("Circular dependency in get_poly!")

            if curr.is_leaf:
                if curr.arg != None:
                    s = SparsePoly(self.n_args)
                    l = [0 for i in range(self.n_args)]
                    l[curr.arg] = 1
                    s[l] = 1
                    if curr is self.root:
                        return s
                    outs[curr] = s
                    curr = track_stack.pop()
                else:
                    s = SparsePoly(self.n_args)
                    s += curr.val
                    if curr is self.root:
                        return s
                    outs[curr] = s
                    curr = track_stack.pop()

            else:
                for op in curr.operands:
                    if op not in outs.keys():
                        track_stack.append(curr)
                        curr = op
                        break
                else:
                    if curr not in outs.keys():

                        if len(curr.operands) != 2:
                            raise ValueError("Node does not have two operands! " + str(len(curr.operands)))

                        outs[curr] = curr.operation.func(
                            outs[curr.operands[0]],
                            outs[curr.operands[1]]
                        )
                        if curr is self.root:
                            break

                    curr = track_stack.pop()

        return outs[self.root]

    def optimize(self):
        change = True
        while change:
            node_copy = self.nodes.copy()
            change = False
            for node in node_copy:
                if sum([1 if op.is_leaf and op.val != None else 0 for op in node.operands]) == 2:
                    better = self.getValNode(node.operation.func(node.operands[0].val, node.operands[1].val), save=False)
                    self.change_input(node, node.operands[0], better)
                    self.removeNode(node, better)
                    change = True

                elif node.operation == OPERATIONS.ADD and sum([1 if op.is_leaf and op.val != None else 0 for op in node.operands]) == 1:

                    for op_node in node.operands:
                        ind_op = 0 if node.operands[0] is op_node else 1

                        if (not op_node.is_leaf) and op_node.operation == OPERATIONS.ADD and sum([1 if op.is_leaf and op.val != None else 0 for op in op_node.operands]) > 0:

                            ind = 0 if op_node.operands[0].is_leaf and op_node.operands[0].val != None else 1
                            better = self.getValNode(node.operation.func(node.operands[1-ind_op].val, op_node.operands[ind].val), save=False)
                            self.removeNode(node, op_node)
                            self.change_input(op_node, op_node.operands[ind], better)
                            change = True
                            break

    def copy(self):
        new_circ = SuperCircuit(self.n_args, costs=self.op_costs)

        other_lib = {}
        for i in range(len(self.arg_leaves)):
            other_lib[self.arg_leaves[i]] = new_circ.getArgNode(i)
        for i in self.val_leaves.keys():
            other_lib[self.val_leaves[i]] = new_circ.getValNode(i)
        
        if self.root.is_leaf:
            return new_circ

        track_stack = []
        curr = self.root
        while True:
            good = True
            if curr.is_leaf:
                other_lib[self.getValNode[curr.val]] = new_circ.getValNode(curr.val)
                curr = track_stack.pop()
                continue

            for op in curr.operands:
                if not op in other_lib.keys():
                    track_stack.append(curr)
                    curr = op
                    good = False
                    break
            if not good:
                continue
            other_ops = [other_lib[op] for op in curr.operands]
            other_lib[curr] = new_circ.addNode(curr.operation, other_ops)
            if curr == self.root:
                return new_circ
            curr = track_stack.pop()


    def getSubCircuit(self):
        pass
        # start with some node

        # keep set of taken nodes, input nodes, output nodes

        # until limit:
            # take node from inputs/outputs and add to taken
                # if that node was input, its inputs become inputs
                # if that node was output, its outputs become outputs
        
        # create new SuperCircuit that has same number of inputs as this

    def updateSubCircuit(self):
        pass

        # somehow take all of the nodes that were in the circuit and replace them in main


def main():
    s = SuperCircuit(1)

    x = s.getArgNode(0)
    one = s.getValNode(1)
    two = s.getValNode(2)

    s.addNode(OPERATIONS.MULT, [x, two])
    print(s.evaluate([1]))
    s.change_operation(s.root, OPERATIONS.ADD)
    print(s.evaluate([1]))
    s.undo()
    print(s.evaluate([1]))

    s_2 = s.copy()
    print(s_2.get_poly())

if __name__ == '__main__':
    main()