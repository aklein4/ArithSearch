
from dataclasses import dataclass
from lib2to3.pytree import Leaf
from circuit import OPERATIONS, Node, LeafNode, OpInfo, treequals
import numpy as np
import sys
import csv

@dataclass
class SmartNode:
    key: np.ndarray
    poly: np.ndarray
    op: OpInfo
    cost: int
    is_leaf: bool
    arg: bool
    val: int
    op_a: int
    op_b: int
    ind: int
    order: int

class NodeLib:
    def __init__(self, filename):
        self.library = []
        with open(filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, dialect='excel')
            header = True
            for row in spamreader:
                if header:
                    header = False
                    continue
                if row[2] == "":
                    if row[1] == 'x':
                        self.library.append({'p': row[1], 'leaf': True, "val": None, "op": None, "op_a": None, "op_b": None})
                    else:
                        self.library.append({'p': row[1], 'leaf': True, "val": int(row[1]), "op": None, "op_a": None, "op_b": None})
                else:
                    self.library.append({
                        'p': row[1], 'leaf': False, "val": None,
                        "op": OPERATIONS.ADD if row[2]=="ADD" else OPERATIONS.MULT,
                        "op_a": int(row[3]), "op_b": int(row[4])
                    })
    
    def __getitem__(self, item):
        d = self.library[item]
        if d["leaf"]:
            return LeafNode(d["val"]), d['p']
        return Node(d["op"], self[d["op_a"]][0], self[d["op_b"]][0]), d['p']

class SmartForceInst:

    def __init__(self, constraints: np.ndarray):
        self.constraints = np.zeros_like(constraints)
        for i in range(self.constraints.shape[0]):
            self.constraints[i] = max(constraints[i:])
        self.key_size = self.constraints.shape[0]
        self.max_order = self.key_size-1
        self.upper_bounds = self.get_poly_key(constraints)
        
        self.lib = {}
        self.ind_lib = {}
        self.curr_ind = 0
        self.mat = None
        # TODO: Prevent repeated trees in the matrix step
        self.prev_inds = set()
        self.curr_depth = 0

        arg_key = self._get_arg_key()
        this_ind = self._save_node(arg_key, OPERATIONS.ADD, -1, -1, cost=0, is_leaf=True, arg=True, val=-100000)
        self.prev_inds.add(this_ind)

        construct_keys = [np.array(arg_key)]
        for n in range(1, max(constraints)+1):
            n_key = self._get_const_key(n)
            construct_keys.append(np.array(n_key))
            this_ind = self._save_node(n_key, OPERATIONS.ADD, -1, -1, cost=0, is_leaf=True, arg=False, val=n)
            self.prev_inds.add(this_ind)

        self.mat = np.stack(construct_keys)

    def _get_arg_key(self):
        return tuple([x for x in range(self.key_size)])
    def _get_const_key(self, k):
        return tuple([k for _ in range(self.key_size)])

    def get_func_key(self, func):
        return tuple([func(x) for x in range(self.key_size)])
    def get_poly_key(self, poly):
        return tuple([sum([poly[i]*(x**i) for i in range(poly.shape[0])]) for x in range(self.key_size)])

    def smart2tree(self, smart: SmartNode):
        if smart.is_leaf:
            return LeafNode(None if smart.arg else smart.val)
        else:
            return Node(smart.op, self.smart2tree(self.ind_lib[smart.op_a]), self.smart2tree(self.ind_lib[smart.op_b]))

    def poly_op(self, op, p_1, p_2):
        if p_1.shape != p_2.shape:
            raise ValueError('Polynomials must be same shape to operate on!')
        if p_1.ndim != 1:
            raise ValueError('Polynomials must be 1 dimensional!')

        if op == OPERATIONS.ADD:
            return p_1+p_2

        new_poly = np.zeros_like(p_1)
        for i in range(p_1.shape[0]):
            part = p_1[i] * p_2
            new_poly[i:] += part[:p_1.shape[0]-i]
        return new_poly

    def _save_node(self, key, op, op_a, op_b, cost=1, is_leaf=False, arg=False, val=-100000000):
        """
        TODO:
        1. Track previous dependencies to account for caching
         - keep a set of the indexes of every subtree (including itself, excluding leaf nodes)
         - this set for a new tree is its own index, union set_op_a, union set_op_b
         - the cost for the new tree is the cardinality of its set
        2. Keep track of ALL found trees for given key
         - when an optimal tree is found, a dependency of another possiblity could change making itself smaller
         - keep list of every possible tree that could produce a given tree
         - find a way to prune his list to only keep 'strictly better' trees
        """

        if not is_leaf:
            cost += self.ind_lib[op_a].cost + self.ind_lib[op_b].cost

        # not best found
        if key in self.lib and self.lib[key].cost <= cost:
            return -1

        # too high of polynomial
        new_order = 0
        if is_leaf:
            if arg:
                new_order = 1
            else:
                new_order = 0
        else:
            new_order = max(self.ind_lib[op_a].order, self.ind_lib[op_b].order)
            if op == OPERATIONS.MULT:
                new_order = self.ind_lib[op_a].order + self.ind_lib[op_b].order
            if new_order > self.max_order:
                return -1
        
        # checking coefficient constraints
        new_poly = np.zeros([self.key_size])
        if is_leaf:
            if arg:
                new_poly[1] = 1
            else:
                new_poly[0] = val
        else:
            new_poly = self.poly_op(op, self.ind_lib[op_a].poly, self.ind_lib[op_b].poly)
            for i in range(new_poly.shape[0]):
                if new_poly[i] > self.constraints[i]:
                    return -1

        # create this node
        my_node = SmartNode(np.array(key), new_poly, op, cost, is_leaf, arg, val, op_a, op_b, -1, new_order)

        if key in self.lib:
            # replace
            my_node.ind = self.lib[key].ind
        else:
            # new
            my_node.ind = self.curr_ind
            self.curr_ind += 1

        self.ind_lib[my_node.ind] = my_node
        self.lib[key] = my_node
        return my_node.ind

    def search(self, max_depth=None, verbose=False):
        if verbose and max_depth == None:
            pred = 1
            for i in range(self.constraints.shape[0]):
                pred *= self.constraints[i]
            print("Predicted Computation Time:", 15.3*round((pred**2)/65535), 's\n')

        while max_depth == None or self.curr_depth < max_depth:
            self.curr_depth += 1
            if verbose:
                print(" --- Iteration", self.curr_depth, "---")
                sys.stdout.write("Searching... ")
                sys.stdout.flush()

            to_add_to_mat = []
            to_add_ind_offset = self.mat.shape[0]
            new_inds = set()

            found = 0
            updated = 0

            # iterate through all depth-1 trees
            msg = ""
            place=1
            for t_ind in self.prev_inds:
                if verbose:
                    if place % max(1, len(self.prev_inds)//100) == 0:
                        erase_msg = ""
                        for _ in range(len(msg)):
                            erase_msg += '\b'
                        for _ in range(len(msg)):
                            erase_msg += ' '
                        for _ in range(len(msg)):
                            erase_msg += '\b'
                        new_msg = str(place)+'/'+str(len(self.prev_inds))
                        sys.stdout.write(erase_msg+new_msg)
                        sys.stdout.flush()
                        msg = new_msg
                    place += 1

                t_key = self.mat[t_ind]

                added_mat = np.add(t_key, self.mat)
                for added_ind in range(added_mat.shape[0]):
                    kickback = self._save_node(
                        tuple(added_mat[added_ind]), OPERATIONS.ADD,
                        t_ind, added_ind
                    ) # - to_add_ind_offset
                    if kickback >= 0:
                        if kickback  - to_add_ind_offset >= 0:
                            if kickback - to_add_ind_offset < len(to_add_to_mat):
                                to_add_to_mat[kickback - to_add_ind_offset] = added_mat[added_ind]
                            else:
                                to_add_to_mat.append(added_mat[added_ind])
                                found += 1
                        else:
                            if kickback not in new_inds:
                                updated += 1
                        new_inds.add(kickback)
                    # if kickback >= 0 and kickback < len(to_add_to_mat):
                    #     to_add_to_mat[kickback] = added_mat[added_ind]
                    #     new_inds.add(kickback+to_add_ind_offset)
                    #     updated += 1
                    # if kickback >= len(to_add_to_mat):
                    #     to_add_to_mat.append(added_mat[added_ind])
                    #     new_inds.add(kickback+to_add_ind_offset)
                
                multed_mat = np.multiply(t_key, self.mat)
                for multed_ind in range(multed_mat.shape[0]):
                    kickback = self._save_node(
                        tuple(multed_mat[multed_ind]), OPERATIONS.MULT,
                        t_ind, multed_ind
                    ) # - to_add_ind_offset
                    if kickback >= 0:
                        if kickback  - to_add_ind_offset >= 0:
                            if kickback - to_add_ind_offset < len(to_add_to_mat):
                                to_add_to_mat[kickback - to_add_ind_offset] = multed_mat[multed_ind]
                            else:
                                to_add_to_mat.append(multed_mat[multed_ind])
                                found += 1
                        else:
                            if kickback not in new_inds:
                                updated += 1
                        new_inds.add(kickback)
                    # if kickback >= 0 and kickback < len(to_add_to_mat):
                    #     to_add_to_mat[kickback] = multed_mat[multed_ind]
                    #     new_inds.add(kickback+to_add_ind_offset)
                    # if kickback >= len(to_add_to_mat):
                    #     to_add_to_mat.append(multed_mat[multed_ind])
                    #     new_inds.add(kickback+to_add_ind_offset)
            
            if verbose:
                print('\nNew Polynomials:', found)
                print('Updated Polynomials:', updated)
                print('New Total:', len(self.lib.keys()))
                print(' ')

            self.prev_inds = new_inds
            if len(to_add_to_mat) > 0:
                self.mat = np.concatenate([self.mat, np.stack(to_add_to_mat)])
            if len(self.prev_inds) == 0:
                return

    def size(self):
        return len(self.ind_lib.keys())

    def save(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            spamwriter.writerow(['index', 'polynomial', 'operation', 'operand 1', 'operand 2'])
            for ind in range(self.curr_ind):
                node = self.ind_lib[ind]
                p = ""
                for i in range(node.poly.shape[0]):
                    if node.poly[i] != 0:
                        if i == 0:
                            p = str(round(node.poly[i]))
                        elif i == 1:
                            p = ("" if node.poly[i]==1 else str(round(node.poly[i])))+'x' + ("+" if p != "" else "") + p
                        else:
                            p = ("" if node.poly[i]==1 else str(round(node.poly[i])))+'x^'+str(i) + ("+" if p != "" else "") + p
                spamwriter.writerow([
                    ind, p,
                    node.op.name if not node.is_leaf else "",
                    node.op_a if not node.is_leaf else "",
                    node.op_b if not node.is_leaf else ""
                ])

def main():
    inst = SmartForceInst(np.array([3, 3, 3, 3, 3, 3, 2, 1]))

    inst.search(verbose=True)
    inst.save("test.csv")

    trees = NodeLib("test.csv")
    while True:
        com = input("Index (enter to quit): ")
        if com == "":
            exit()
        tree, p = trees[int(com)]
        print('\n'+p+'\n')
        tree.show()
        print('')

    # print(inst.size())
    #print(inst.ind_lib)
    # for tree in inst.lib.keys():
    #     inst.smart2tree(inst.lib[tree]).show()
    #print(inst.lib[inst.get_func_key(lambda x: x**3 + x**2 + x +1)].ind)
    #inst.smart2tree(inst.lib[inst.get_func_key(lambda x: x**3 + x**2 + x +1)]).show()

if __name__ == '__main__':
    main()