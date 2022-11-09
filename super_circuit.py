
from circuit import OPERATIONS, OpInfo

class SuperNode:

    def __init__(self, operands=None, operation=None, is_leaf=False, arg=None, val=None):
        if operands != None and type(operands) is not set:
            raise ValueError("Operands must be presented as a set!")

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
        self.outputs.remove(out)


    def change_input(self, old_input, new_input):
        if self.is_leaf:
            raise ValueError("Cannot change the input to a leaf!")

        # check to avoid circular dependencies
        if new_input is self or self in new_input.depends_on:
            return False

        old_input.remove_output(self)
        new_input.set_output(self)

        self.operands.remove(old_input)
        self.operands.add(new_input)

        self._update_depends()
        return True
        
    def _update_depends(self):

        new_depends = self.operands.copy()
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

        # hold const leaves in dict with val -> node
        self.val_leaves = {}

        # hold arg leaves in list
        self.arg_leaves = [
            SuperNode(is_leaf=True, arg=i) for i in range(n_args)
        ]

        # get root node of tree
        self.root = None

        # keep track of the number of circle nodes that depend on a given one
        self.num_dep = {}
        # keep track of which cicle nodes have multiple outputs
        self.multi_out = set()

        # keep track of how mush operations cost
        self.op_costs = costs.copy()
        # keep track of how much this tree costs
        self.cost = 0

        # the action that undoes the last action
        self.undo_action = None


    def getValNode(self, val):
        """
        :return: a leaf node of the given value. Creates one if it doesn't already exist
        """
        if val in self.val_leaves:
            return self.val_leaves[val]
        else:
            new_node = SuperNode(is_leaf=True, val=val)
            self.val_leaves[val] = new_node
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


    def addNode(self, operation, operands):
        """
        Add a new node to the tree
        :arg operation: the operation that this node performs
        :arg operands: Iterable of operands that this will take as inputs
        """
        # TODO: Check if this violates circular dependency

        # create the new node
        new_node = SuperNode(operation=operation, operands=operands)
        # add it to the register
        self.nodes.add(new_node)
        self.num_dep[new_node] = 0
        self.cost += self.op_costs[operation]

        # add this dependency to its child nodes
        for op in operands:
            if not op.is_leaf:
                self.num_dep[op] += 1
                if self.num_dep[op] > 1:
                    self.multi_out.add(op)

        # this is the new root node if it depends on the old root
        for op in operands:
            if self.root == None or op is self.root:
                self.root = new_node
                break

    def _change_operation_wrap(self, args):
        return self.change_operation(args[0], args[1])
    def change_operation(self, node, operation):
        if node.is_leaf:
            raise ValueError("Cannot change the operation of a leaf node!")
        
        if node.operation == operation:
            return False

        self.undo_action = {"func": self._change_operation_wrap, "args": (node, node.operation)}
        node.operation = operation
        return True


    def _change_input_wrap(self, args):
        return self.change_input(args[0], args[1], args[2])
    def change_input(self, node, old_input, new_input):
        succ = node.change_input(old_input, new_input)

        if succ:
            self.undo_action = {"func": self._change_input_wrap, "args": (node, new_input, old_input)}

            if not old_input.is_leaf:

                self.num_dep[old_input] -= 1
                if self.num_dep[old_input] == 1:
                    self.multi_out.remove(old_input)

            if not new_input.is_leaf:
                self.num_dep[new_input] += 1
                if self.num_dep[new_input] > 1:
                    self.multi_out.add(new_input)

        return succ


    def undo(self):
        if self.undo_action == None:
            return
        
        self.undo_action["func"](self.undo_action["args"])

        self.undo_action == None


    def clean(self):
        to_remove = set()
        for n in self.num_dep.keys():
            if self.num_dep[n] == 0:
                to_remove.add(n)
        
        for n in to_remove:
            self.nodes.remove(n)
            self.num_dep.pop(n)
        
        return len(to_remove)


    def evaluate(self, args):

        outs = {}
        deps = {}
        track_stack = []
        curr = self.root

        max_stack_size = 1.5*(len(self.nodes) + len(self.arg_leaves) + len(self.val_leaves))
        while True:
            if len(track_stack) > max_stack_size:
                raise RuntimeError("Evaluation stack depth exceeds limit, circuit likely contains circular dependency.")

            if curr.is_leaf:
                if curr.arg != None:
                    outs[curr] = args[curr.arg]
                    deps[curr] = len(curr.outputs)
                    curr = track_stack.pop()
                else:
                    outs[curr] = curr.val
                    deps[curr] = len(curr.outputs)
                    curr = track_stack.pop()

            else:
                for op in curr.operands:
                    if op not in outs.keys():
                        track_stack.append(curr)
                        curr = op
                        break
                else:
                    if curr not in outs.keys():

                        op_list = list(curr.operands)
                        if len(op_list) != 2:
                            raise ValueError("Node does not have two operands!")

                        outs[curr] = curr.operation.func(
                            outs[op_list[0]],
                            outs[op_list[1]]
                        )
                        if curr is self.root:
                            break

                        deps[curr] = len(curr.outputs)
                        for op in curr.operands:
                            deps[op] -= 1
                            if deps[op] == 0:
                                deps.pop(op)
                                outs.pop(op)

                    curr = track_stack.pop()

        return outs[self.root]


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

    s.addNode(OPERATIONS.MULT, set([x, one]))
    print(s.evaluate([1]))
    print(s.change_operation(s.root, OPERATIONS.ADD))
    print(s.evaluate([1]))
    s.undo()
    print(s.evaluate([1]))

if __name__ == '__main__':
    main()