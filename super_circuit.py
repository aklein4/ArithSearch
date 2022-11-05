
from circuit import OPERATIONS, OpInfo

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
        self.is_leaf = is_leaf # whether this is leaf
        if self.is_leaf:
            # if leaf, then either const val or arg
            self.arg = arg # number representing index of input
            self.val = val
        else:
            # else do operation
            self.operation = operation
            self.operands = operands

            for operand in operands:
                operand.set_output(self)
    
    def set_output(self, out):
        # set node as output of self
        self.outputs.add(out)

    def remove_output(self, out):
        self.outputs.remove(out)


class SuperCircuit:

    def __init__(self, n_args):
        # number of inputs that this tree takes
        self.n_args = n_args

        # hold nodes in some data type
        self.nodes = set()

        # hold all nodes that point to a box
        self.end_nodes = set()

        # hold const leaves in dict with val -> node
        self.val_leaves = {}

        # hold arg leaves in list
        self.arg_leaves = [
            SuperNode(is_leaf=True, arg=i) for i in range(n_args)
        ]

        # get root node of tree
        self.root = None

        # TODO: track dependency sets to avoid circular dependancy

    def getValNode(self, val):
        if val in self.val_leaves:
            return self.val_leaves[val]
        else:
            new_node = SuperNode(is_leaf=True, val=val)
            self.val_leaves[val] = new_node
            return new_node

    def getArgNode(self, arg):
        return self.arg_leaves[arg]

    def addNode(self, operation, operands):
        new_node = SuperNode(operation=operation, operands=operands)
        self.nodes.add(new_node)

        for op in operands:
            if self._isPointer(op):
                self.end_nodes.add(new_node)
                break
        
        for op in operands:
            if self.root == None or op is self.root:
                self.root = new_node
                break

    def _isPointer(self, edge):
        return edge in self.arg_leaves or edge in self.val_leaves

    def evaluate(self, args):

        outs = {}
        deps = {}
        track_stack = []
        curr = self.root

        while True:
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
                if curr.operands[0] not in outs.keys():
                    track_stack.append(curr)
                    curr = curr.operands[0]
                elif curr.operands[1] not in outs.keys():
                    track_stack.append(curr)
                    curr = curr.operands[1] 
                else:
                    if curr not in outs.keys():
                        outs[curr] = curr.operation.func(
                            outs[curr.operands[0]],
                            outs[curr.operands[1]]
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

    s.addNode(OPERATIONS.ADD, [x, one])

    print(s.evaluate([7]))


if __name__ == '__main__':
    main()