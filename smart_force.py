
from dataclasses import dataclass
from circuit import OPERATIONS, Node, LeafNode, OpInfo, treequals
import numpy as np
import sys
import csv


# constraints to satisfy when running as main
# each v[i] represents the maximum coefficient of x^i
# time complexity is ~ (product of all constraints)^2, assuming that v[i+1] <= v[i]
CONSTRAINTS_TO_USE = [4, 4, 4, 4]


@dataclass
class SmartNode:
    """
    Contains information about a polynomial computation tree
    """
    key: np.ndarray # vector in R^n representing the polynomial's point valeus
    poly: np.ndarray # vecotor in R^n+1 representing the polynomial's coefficients
    op: OpInfo # operation performed at top of tree
    cost: int # total cost (# operations) of tree
    is_leaf: bool # whether tree is leaf node
    arg: bool # if is_leaf, whether node is 'x' argument
    val: int # if is_leaf and not arg, const value of node
    op_a: int # index of operand a
    op_b: int # index of operand b
    ind: int # index of self
    order: int # order of self's polynomial


class SmartForceInst:
    """
    Generates the arithmatic circuit trees for all polynomials that satisfies the given constraints.
    All results are guarenteed to be globally optimal within the problem's assumptions:
     - only positive values
     - strictly tree-shaped circuits without cross-chaching
    """

    def __init__(self, constraints: np.ndarray):
        """
        :param constraints: numpy vector with v_i representing max coefficient of x^i (coefficients beyong length assumed to be zero)
        """
        # reconfigure constraints to match internal use
        self.constraints = np.zeros_like(constraints)
        for i in range(self.constraints.shape[0]):
            self.constraints[i] = max(constraints[i:])
        # n points to describe  nth degree polynomial
        self.key_size = self.constraints.shape[0]
        # maximum order of the polynomial
        self.max_order = self.key_size-1
        # key points of polynomial at upper bound of all constraints
        self.upper_bounds = self.get_poly_key(constraints)

        self.lib = {} # dict mapping key -> node
        # TODO: turn ind_lib to list
        self.ind_lib = {} # dict mapping index -> node
        self.curr_ind = 0 # dict mapping index -> node
        self.mat = None # matrix containing rows of every found key
        # TODO: Prevent repeated trees in the matrix step
        self.prev_inds = set() # set of indexes that were modified in previous step
        self.curr_iteration = 0 # current iteration of search

        # insert argument leaf
        arg_key = self._get_arg_key()
        this_ind = self._save_node(arg_key, OPERATIONS.ADD, -1, -1, cost=0, is_leaf=True, arg=True, val=-100000)
        self.prev_inds.add(this_ind)

        # insert constant leaves up to max constraint
        construct_keys = [np.array(arg_key)]
        for n in range(1, max(constraints)+1):
            n_key = self._get_const_key(n)
            construct_keys.append(np.array(n_key))
            this_ind = self._save_node(n_key, OPERATIONS.ADD, -1, -1, cost=0, is_leaf=True, arg=False, val=n)
            self.prev_inds.add(this_ind)

        # save new leaves to mat
        self.mat = np.stack(construct_keys)


    def _get_arg_key(self):
        # get key of argument
        return tuple([x for x in range(self.key_size)])
    def _get_const_key(self, k):
        # get key of constant
        return tuple([k for _ in range(self.key_size)])
    def get_func_key(self, func):
        # get key of function
        return tuple([func(x) for x in range(self.key_size)])
    def get_poly_key(self, poly):
        # get key of polynomial described by coefficients
        return tuple([sum([poly[i]*(x**i) for i in range(poly.shape[0])]) for x in range(self.key_size)])


    def smart2tree(self, smart: SmartNode):
        """
        Convert a SmartNode tree into a regular Node tree (ex. for printing)
        :param smart: SmartNode to convert
        """
        if smart.is_leaf:
            # return leaf
            return LeafNode(None if smart.arg else smart.val)
        else:
            # return recurive combo of children
            return Node(smart.op, self.smart2tree(self.ind_lib[smart.op_a]), self.smart2tree(self.ind_lib[smart.op_b]))


    def poly_op(self, op, p_1, p_2):
        """
        Do operation on polynomials described by numpy vector of coefficients
        :param op: OpInfo describing ADD or MULT
        :param p_1: coefficients of first polynomial
        :param p_2: coefficients of second polynomial
        """

        # check sizes
        if p_1.shape != p_2.shape:
            raise ValueError('Polynomials must be same shape to operate on!')
        if p_1.ndim != 1:
            raise ValueError('Polynomials must be 1 dimensional!')

        # addition just adds coefficients
        if op == OPERATIONS.ADD:
            return p_1+p_2

        # multiplication uses foil method
        # TODO: find faster way to do this
        new_poly = np.zeros_like(p_1)
        for i in range(p_1.shape[0]):
            part = p_1[i] * p_2
            new_poly[i:] += part[:p_1.shape[0]-i]
        return new_poly


    def _save_node(self, key, op, op_a, op_b, cost=1, is_leaf=False, arg=False, val=-100000000):
        """
        Check if a node described by the inputs should be saved, and if so then do it (not added to mat).
        :param key: key describing new node
        :param op: top-level operation describing node
        :param op_a: index of first operand
        :param op_b: index of second operand
        :param cost: Cost of top-level operation
        :param is_leaf: Whether this is a leaf node
        :param arg: If leaf, whether takes 'x' argument
        :param val: If leaf and not arg, value of constant
        :return -1 if node not used, else index of added/updated node
        """

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

        # calculate cost of new node
        if not is_leaf:
            cost += self.ind_lib[op_a].cost + self.ind_lib[op_b].cost

        # seen this key before, and this is not better than that one
        if key in self.lib and self.lib[key].cost <= cost:
            return -1

        # check if the order of this node violates constraints
        new_order = 0
        if is_leaf:
            if arg:
                # order of 'x' is 1
                new_order = 1
            else:
                # order of 'a' is 0
                new_order = 0
        else:
            # ADD takes max order of children
            new_order = max(self.ind_lib[op_a].order, self.ind_lib[op_b].order)
            if op == OPERATIONS.MULT:
                # MULT adds orders of children
                new_order = self.ind_lib[op_a].order + self.ind_lib[op_b].order
        if new_order > self.max_order:
            # violates constraint
            return -1
        
        # check key point restraints
        for i in range(len(self.upper_bounds)):
            if key[i] > self.upper_bounds[i]:
                # violates constraint
                return -1

        # check coefficient constraints
        new_poly = np.zeros([self.key_size])
        if is_leaf:
            if arg:
                # 'x' is [0, 1, ...]
                new_poly[1] = 1
            else:
                # 'a' is [a, 1, ...]
                new_poly[0] = val
        else:
            # combine polynomials of children
            new_poly = self.poly_op(op, self.ind_lib[op_a].poly, self.ind_lib[op_b].poly)
        for i in range(new_poly.shape[0]):
            if new_poly[i] > self.constraints[i]:
                # violates constraint
                return -1

        # create new node
        my_node = SmartNode(np.array(key), new_poly, op, cost, is_leaf, arg, val, op_a, op_b, -1, new_order)

        if key in self.lib:
            # replace in libs
            my_node.ind = self.lib[key].ind
        else:
            # add to libs
            my_node.ind = self.curr_ind
            # next index is one higher
            self.curr_ind += 1

        # save to libs
        self.ind_lib[my_node.ind] = my_node
        self.lib[key] = my_node

        # return updated index
        return my_node.ind


    def search(self, verbose=True, iterative_deepening=True):
        """
        Find optimal computation tree for all polynomials within constraints.
        :param verbode: Whether to print update messages (default: True)
        """

        the_max_order = self.max_order
        for desired_order in range(1, the_max_order+1 if iterative_deepening else 2):
            if verbose:
                print(" ----- ORDER", desired_order, " -----\n")
            if iterative_deepening:
                self.max_order = desired_order

            self.prev_inds = set()
            for i in range(0, self.curr_ind):
                self.prev_inds.add(i)

            # iterate until there is nothing left to do and we return
            while True:

                # iteration header
                self.curr_iteration += 1
                if verbose:
                    print(" --- Iteration", self.curr_iteration, "---")
                    sys.stdout.write("Searching... ")
                    sys.stdout.flush()

                to_add_to_mat = [] # list of keys to add to mat at end of iteration
                to_add_ind_offset = self.mat.shape[0] # difference between to_add_to_mt index and future mat index
                new_inds = set() # keep track of indexes that are updated this iteration

                found = 0 # new polynomials we find
                updated = 0 # polynomials we update

                # verbose stuff
                msg = ""
                place = 1

                # iterate through all indexes that were changed in last iteration
                for t_ind in self.prev_inds:

                    # show progress message
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

                    # get key of current index
                    t_key = self.mat[t_ind]

                    # calculate keys of all trees formed by combining t_key with another known
                    added_mat = np.add(t_key, self.mat)
                    # iterate through all added keys
                    # TODO: Prevent some addition pairs from being seen twice from different t_inds
                    for added_ind in range(added_mat.shape[0]):
                        if iterative_deepening and max(self.ind_lib[added_ind].order, self.ind_lib[t_ind].order) != desired_order:
                            continue
                        # try saving this new node
                        kickback = self._save_node(
                            tuple(added_mat[added_ind]), OPERATIONS.ADD,
                            t_ind, added_ind
                        )
                        if kickback >= 0:
                            # key was used
                            if kickback  - to_add_ind_offset >= 0:
                                # this polynomial wa not known before this iteration
                                if kickback - to_add_ind_offset < len(to_add_to_mat):
                                    # seen before in this iteration
                                    to_add_to_mat[kickback - to_add_ind_offset] = added_mat[added_ind]
                                else:
                                    # never seen before
                                    to_add_to_mat.append(added_mat[added_ind])
                                    found += 1
                            else:
                                # updated old known key
                                if kickback not in new_inds:
                                    updated += 1
                            # save this change for next iteration
                            new_inds.add(kickback)
                    
                    # same as above but with multiplication
                    # TODO: decompose this block into function called for both ADD and MULT
                    multed_mat = np.multiply(t_key, self.mat)
                    for multed_ind in range(multed_mat.shape[0]):
                        if iterative_deepening and self.ind_lib[multed_ind].order + self.ind_lib[t_ind].order != desired_order:
                            continue
                        kickback = self._save_node(
                            tuple(multed_mat[multed_ind]), OPERATIONS.MULT,
                            t_ind, multed_ind
                        )
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
                
                # print info on what we found
                if verbose:
                    print('\nNew Polynomials:', found)
                    print('Updated Polynomials:', updated)
                    print('New Total:', len(self.lib.keys()))
                    print(' ')

                # replace prev inds with ones from this iteration
                self.prev_inds = new_inds

                # add new keys to mat
                if len(to_add_to_mat) > 0:
                    self.mat = np.concatenate([self.mat, np.stack(to_add_to_mat)])
                    
                # if nothing changed, we are done
                if len(self.prev_inds) == 0:
                    break


    def size(self):
        # return size of library
        return len(self.ind_lib.keys())

    def save(self, filename=None):
        """
        Save library to a file that can be read in read_lib
        :param filename: name of file to write to, or constraints of None (default)
        """

        # make filename
        if filename == None:
            filename = ""
            for i in range(self.constraints.shape[0]):
                filename += str(self.constraints[i]) + "-"
            filename = filename[:-1]+".csv"

        # write to csv
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            # header
            spamwriter.writerow(['index', 'polynomial', 'operation', 'operand 1', 'operand 2'])

            # iterate through everything in lib
            for ind in range(self.curr_ind):
                node = self.ind_lib[ind]

                # turn polynomial into string to save
                p = ""
                for i in range(node.poly.shape[0]):
                    if node.poly[i] != 0:
                        if i == 0:
                            # 0th order
                            p = str(round(node.poly[i]))
                        elif i == 1:
                            # first order 'x'
                            p = ("" if node.poly[i]==1 else str(round(node.poly[i])))+'x' + ("+" if p != "" else "") + p
                        else:
                            # any other 'x^n'
                            p = ("" if node.poly[i]==1 else str(round(node.poly[i])))+'x^'+str(i) + ("+" if p != "" else "") + p
                
                # save in accordance with header
                spamwriter.writerow([
                    ind, p,
                    node.op.name if not node.is_leaf else "",
                    node.op_a if not node.is_leaf else "",
                    node.op_b if not node.is_leaf else ""
                ])

def main():
    # init
    inst = SmartForceInst(np.array(CONSTRAINTS_TO_USE), negative=True)

    # execute
    inst.search(verbose=True)
    inst.save()

    print(" --- Program Complete! ---\n")

if __name__ == '__main__':
    main()