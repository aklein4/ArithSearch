
from circuit import OPERATIONS, Node, LeafNode, treequals

class BruteForceInst:

    def __init__(self, n_set):
        self.depth = 0
        self.T = [LeafNode(val=None)]
        self.add_assoc = []
        self.mult_assoc = []
        for n in n_set:
            self.add_assoc.append([])
            self.mult_assoc.append([])
            self.T.append(LeafNode(val=n))
        self.prev_start = 0
    
    def grow_to(self, max_depth, perm_lock=False):
        if self.depth >= max_depth:
            return 0
        
        growth = 0
        while self.depth < max_depth:
            new_T = []
            for op in [OPERATIONS.ADD, OPERATIONS.MULT]:
                seen_perms = set()
                for t in range(self.prev_start, len(self.T)):
                    for tau in range(t+1):
                        new_node = Node(op, self.T[t], self.T[tau])
                        if perm_lock:
                            perm = frozenset([pair for pair in new_node.assocs.items()])
                            if perm not in seen_perms:
                                seen_perms.add(perm)
                                new_T.append(new_node)
                        else:
                            new_T.append(new_node)
            self.prev_start = len(self.T)
            self.T += new_T
            growth += len(new_T)
            self.depth += 1
        return growth

def main():
    inst = BruteForceInst([i for i in range(1, 2)])

    func = lambda x: x**3 + x**2 + 2*x + 1

    for i in range(1, 4):
        g = inst.grow_to(i, perm_lock=False)
        print(i, len(inst.T))
    print("All trees found.")

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
                best_list.append(tree)
            else:
                s = tree.size()
                if best == None or s < best:
                    best = s
                    best_list = [tree]
                elif s == best:
                    best_list.append(tree)
    
    for tree in best_list:
        tree.show()
    print("Tied:", len(best_list))

if __name__ == '__main__':
    main()