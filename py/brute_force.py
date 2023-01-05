
from circuit import OPERATIONS, Node, LeafNode, treequals

class BruteForceInst:

    def __init__(self, n_set):
        """
        :param n_set: Set of constant-valued leaf-nodes to use (ex. [1, 2, 3])
        """
        self.depth = 0 # current depth we've searched to
        self.T = [LeafNode(val=None)] # list of all trees seen
        for n in n_set:
            self.T.append(LeafNode(val=n))
        self.prev_start = 0 # where to start searching on the next depth increase
    
    def grow_to(self, max_depth, perm_lock=False):
        """
        Find all trees up to depth max_depth and store them in self.T
        :param max_depth: Maximum tree depth to search to
        :param perm_lock: Abuse associative property to weed out repeats
        :return The number of new trees found
        """

        # already there
        if self.depth >= max_depth:
            return 0
        
        growth = 0
        # grow to new depth
        while self.depth < max_depth:

            # hold new trees
            new_T = []

            # new loop through possible operations
            for op in [OPERATIONS.ADD, OPERATIONS.MULT]:
                seen_perms = set()

                # loop through possible first operands
                for t in range(self.prev_start, len(self.T)):
                    # loop through possible second operands
                    for tau in range(t+1):
                        new_node = Node(op, self.T[t], self.T[tau])

                        # check if this permutation has been seen
                        if perm_lock:
                            perm = frozenset([pair for pair in new_node.assocs.items()])
                            if perm not in seen_perms:
                                seen_perms.add(perm)
                                new_T.append(new_node)
                        # always keep new tree
                        else:
                            new_T.append(new_node)

            # we have already looped through these trees, don't need to next time
            self.prev_start = len(self.T)

            # save new trees
            self.T += new_T
            growth += len(new_T)

            # iterate to next depth
            self.depth += 1

        # return total found
        return growth

def main():
    # init
    inst = BruteForceInst([i for i in range(1, 2)])

    # define polynomial we are looking for
    func = lambda x: x**3 + x**2 + 2*x + 1

    # grow to new depth
    for i in range(1, 4):
        g = inst.grow_to(i, perm_lock=False)
        print(i, len(inst.T))
    print("All trees found.")

    # search found trees for accuracy and cost
    best = None
    best_list = []
    for i in range(len(inst.T)):
        tree = inst.T[i]
        good = True
        for x in range(6):
            if func(x) != tree.run(x):
                good = False
                break
        if good:
            if False:
                # log all trees that are accurate
                best_list.append(tree)
            else:
                # log only the best (or tied) trees
                s = tree.size()
                if best == None or s < best:
                    best = s
                    best_list = [tree]
                elif s == best:
                    best_list.append(tree)
    
    # show found solutions
    for tree in best_list:
        tree.show()
    print("Tied:", len(best_list))

if __name__ == '__main__':
    main()