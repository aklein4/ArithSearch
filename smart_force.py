
from dataclasses import dataclass
from circuit import OPERATIONS, Node, LeafNode, OpInfo, treequals
import numpy as np
import sys
import csv
import time

# constraints to satisfy when running as main
# each v[i] represents the maximum coefficient of x^i
# time complexity is ~ (product of all constraints)^2, assuming that v[i+1] <= v[i]
CONSTRAINTS_TO_USE = [9, 9, 9, 9, 9, 9]

# data seems to suggest that costs are never greater than 2*order
# this enforces that constraint for pruning (this is confirmed by Horner's algorithm)
ASSUME_UPPER_LIMIT = True

# folder to save data in
SAVE_FOLDER = "./data/"

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
    depth: int


class SmartForceInst:
    """
    Generates the arithmatic circuit trees for all polynomials that satisfies the given constraints.
    All results are guarenteed to be globally optimal within the problem's assumptions:
     - only positive values
     - strictly tree-shaped circuits without cross-chaching
    """

    def __init__(self, constraints: np.ndarray, check_depth=False):
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

        self.bins = {}
        self.check_depth = check_depth

        # insert argument leaf
        arg_key = self._get_arg_key()
        this_ind, delta = self._save_node(arg_key, OPERATIONS.ADD, -1, -1, cost=0, is_leaf=True, arg=True, val=-100000)
        self.prev_inds.add(this_ind)
        self.put_in_bin(this_ind, 0)

        for n in range(1, max(constraints)+1):
            n_key = self._get_const_key(n)
            this_ind, delta = self._save_node(n_key, OPERATIONS.ADD, -1, -1, cost=0, is_leaf=True, arg=False, val=n)
            self.prev_inds.add(this_ind)
            self.put_in_bin(this_ind, 0)


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


    def put_in_bin(self, ind, new_cost, old_cost=None):
        if new_cost == old_cost:
            return
        if old_cost != None:
            self.bins[old_cost].remove(ind)
        if new_cost not in self.bins.keys():
            self.bins[new_cost] = set()
        self.bins[new_cost].add(ind)


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

        new_depth = 0
        if not is_leaf:
            new_depth = 1+max(self.ind_lib[op_a].depth, self.ind_lib[op_b].depth)

        # seen this key before, and this is not better than that one
        if key in self.lib and self.lib[key].cost < cost:
            return -1, 0
        if key in self.lib and self.lib[key].cost == cost:
            if self.check_depth and new_depth < self.lib[key].depth:
                pass
            else:
                return -1, 0


        old_cost = None
        if key in self.lib:
            old_cost = self.lib[key].cost

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
        if new_order > self.max_order or (ASSUME_UPPER_LIMIT and cost > 2*new_order):
            # violates constraint
            return -1, 0
        
        # check key point restraints
        for i in range(len(self.upper_bounds)):
            if key[i] > self.upper_bounds[i]:
                # violates constraint
                return -1, 0

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
                return -1, 0

        # create new node
        my_node = SmartNode(np.array(key), new_poly, op, cost, is_leaf, arg, val, op_a, op_b, -1, new_order, new_depth)

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
        return my_node.ind, cost-old_cost if old_cost != None else None


    def search(self, verbose=True, iterative_deepening=True, save_progress=False):
        """
        Find optimal computation tree for all polynomials within constraints.
        :param verbode: Whether to print update messages (default: True)
        :param iterative_deepening: Build library with increasing order, seems to run much faster
        """

        the_max_order = self.max_order
        # loop through iterative polynomials if iterative_deepening
        for desired_order in range(1, the_max_order+1 if iterative_deepening else 2):
            if verbose:
                print(" ----- ORDER", desired_order, " -----\n")
            if iterative_deepening:
                self.max_order = desired_order

            # put all previous polynomials into the set to try
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

                created_inds = {} # list of keys to add to mat at end of iteration
                replaced_inds = {}
                to_add_ind_offset = self.curr_ind # difference between to_add_to_mt index and future mat index
                new_inds = set() # keep track of indexes that are updated this iteration

                found = 0 # new polynomials we find
                updated = 0 # polynomials we update

                # verbose stuff
                msg = ""
                place = 1
                last_time = time.time_ns()

                # iterate through all indexes that were changed in last iteration
                for t_ind in self.prev_inds:

                    # show progress message
                    if verbose:
                        new_time = time.time_ns()
                        if place == 1 or (new_time-last_time)*1e-9 >= 0.25:
                            last_time = new_time
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
                    t_key = self.ind_lib[t_ind].key

                    # calculate keys of all trees formed by combining t_key with another known
                    if iterative_deepening and self.ind_lib[t_ind].order != desired_order:
                        pass
                    else:
                        # iterate through all added keys
                        # TODO: Prevent some addition pairs from being seen twice from different t_inds
                        for other_cost in range(0, (2*desired_order)-self.ind_lib[t_ind].cost):
                            for added_ind in self.bins.get(other_cost, []):
                                added_mat = np.add(np.array(t_key), np.array(self.ind_lib[added_ind].key))
                                if iterative_deepening and max(self.ind_lib[added_ind].order, self.ind_lib[t_ind].order) != desired_order:
                                    # iterative deepening skip
                                    continue
                                # try saving this new node
                                kickback, delta = self._save_node(
                                    tuple(added_mat), OPERATIONS.ADD,
                                    t_ind, added_ind
                                )
                                if kickback >= 0:
                                    # key was used
                                    if kickback  - to_add_ind_offset >= 0:
                                        # this polynomial was not known before this iteration
                                        if kickback in created_inds.keys():
                                            # seen before in this iteration
                                            created_inds[kickback] = 1+self.ind_lib[t_ind].cost+other_cost
                                        else:
                                            # never seen before
                                            created_inds[kickback] = 1+self.ind_lib[t_ind].cost+other_cost
                                            found += 1
                                    else:
                                        if kickback in replaced_inds.keys():
                                            # seen before in this iteration
                                            replaced_inds[kickback][0] = 1+self.ind_lib[t_ind].cost+other_cost
                                        else:
                                            # never seen before
                                            created_inds[kickback] = (1+self.ind_lib[t_ind].cost+other_cost, 1+self.ind_lib[t_ind].cost+other_cost - delta)
                                            updated += 1
                                    # save this change for next iteration
                                    new_inds.add(kickback)
                    
                    # same as above but with multiplication
                    # TODO: decompose this block into function called for both ADD and MULT
                    if iterative_deepening and self.ind_lib[t_ind].order > (desired_order / 2) and self.ind_lib[t_ind].order != desired_order:
                        pass
                    else:
                        for other_cost in range(0, (2*desired_order)-self.ind_lib[t_ind].cost):
                            for multed_ind in self.bins.get(other_cost, []):
                                if iterative_deepening and self.ind_lib[multed_ind].order + self.ind_lib[t_ind].order != desired_order:
                                    # iterative deepening skip
                                    continue
                                multed_mat = np.multiply(np.array(t_key), np.array(self.ind_lib[multed_ind].key))
                                kickback, delta = self._save_node(
                                    tuple(multed_mat), OPERATIONS.MULT,
                                    t_ind, multed_ind
                                )
                                if kickback >= 0:
                                    # key was used
                                    if kickback  - to_add_ind_offset >= 0:
                                        # this polynomial was not known before this iteration
                                        if kickback in created_inds.keys():
                                            # seen before in this iteration
                                            created_inds[kickback] = 1+self.ind_lib[t_ind].cost+other_cost
                                        else:
                                            # never seen before
                                            created_inds[kickback] = 1+self.ind_lib[t_ind].cost+other_cost
                                            found += 1
                                    else:
                                        if kickback in replaced_inds.keys():
                                            # seen before in this iteration
                                            replaced_inds[kickback][0] = 1+self.ind_lib[t_ind].cost+other_cost
                                        else:
                                            # never seen before
                                            created_inds[kickback] = (1+self.ind_lib[t_ind].cost+other_cost, 1+self.ind_lib[t_ind].cost+other_cost - delta)
                                            updated += 1
                                    # save this change for next iteration
                                    new_inds.add(kickback)
                
                # print info on what we found
                if verbose:
                    print('\nNew Polynomials:', found)
                    print('Updated Polynomials:', updated)
                    print('New Total:', len(self.lib.keys()))
                    print(' ')

                # replace prev inds with ones from this iteration
                self.prev_inds = set()
                for ind in new_inds:
                    if self.ind_lib[ind].cost <= (2*desired_order)-1:
                        self.prev_inds.add(ind)

                # add new keys to mat
                for ind in created_inds.keys():
                    self.put_in_bin(ind, created_inds[ind])
                for ind in replaced_inds.keys():
                    self.put_in_bin(ind, replaced_inds[ind][0], replaced_inds[ind][1])
                    
                # if nothing changed, we are done
                if len(self.prev_inds) == 0:
                    break
            
            if save_progress:
                if verbose:
                    sys.stdout.write("Saving... ")
                    sys.stdout.flush()
                self.save()
                if verbose:
                    sys.stdout.write("done. \n\n")


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
            for i in range(self.max_order + 1):
                filename += str(self.constraints[i]) + "-"
            filename = filename[:-1]+".csv"

        # write to csv
        with open(SAVE_FOLDER+filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            # header
            spamwriter.writerow(['index', 'polynomial', 'operation', 'operand 1', 'operand 2', 'cost', 'order', 'depth'])

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
                    node.op_b if not node.is_leaf else "",
                    node.cost, node.order, node.depth
                ])

def main():
    # init
    inst = SmartForceInst(np.array(CONSTRAINTS_TO_USE), check_depth=True)

    # execute
    start_time = time.time()
    inst.search(verbose=True, iterative_deepening=True, save_progress=False)
    print(" --- Search Complete! (" + str(round(time.time()-start_time, 1)) + " s) ---\n")

    # save final output
    sys.stdout.write("Saving... ")
    sys.stdout.flush()
    inst.save("depth.csv")
    print("done.")

if __name__ == '__main__':
    main()