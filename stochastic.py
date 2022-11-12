
import numpy as np
import numpy.polynomial as poly
import random
import itertools
import sys
import time
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sparse_poly import  SparsePoly, poly_MSE, LETTERS

from super_circuit import SuperCircuit
from circuit import OPERATIONS

import warnings
warnings.simplefilter('ignore', np.RankWarning)

"""
Framework for potential stochastic search algorithm.
"""

def random_op():
    return OPERATIONS.ADD if random.random() >= 0.5 else OPERATIONS.MULT

def not_my_op(me):
    if me.operation is OPERATIONS.ADD:
        return OPERATIONS.MULT
    return OPERATIONS.ADD

class SearchEngine:

    def __init__(self, target: SparsePoly, costs={OPERATIONS.MULT: 1, OPERATIONS.ADD: 1}, n_vals=10):
        self.circuit = SuperCircuit(target.n, costs=costs)
        self.target = target.copy()
        self.n_args = target.n

        for i in range(1, n_vals+1):
            self.circuit.getValNode(i)
            # self.circuit.getValNode(-i)

    def get_samples(self, samples_per_arg):
        size = samples_per_arg**self.n_args
        samples = np.zeros((self.n_args, size))

        seen = set()
        iters = itertools.combinations_with_replacement([i for i in range(samples_per_arg)], self.n_args)
        ind = 0
        for it in iters:
            perms = itertools.permutations(it)
            for perm in perms:
                if perm not in seen:
                    for arg in range(self.n_args):
                        samples[arg][ind] = perm[arg]
                    ind += 1
                    seen.add(perm)

        if len(samples.shape) == 1:
            samples = np.expand_dims(samples, 0)
        return samples.astype(np.int64)

    # def get_target_output(self, samples):
    #     return self.target_func(samples)

    def search(self, max_iters, C_cost, C_acc, temp, clean_freq=100, verbose=True, return_on_first=False, L=0.5):

        # things relating to the target
        samples = self.get_samples(10)
        # target = np.pad(self.target, (0, samples.shape[1] - self.target.size + 1))

        # keep track of the best solution that we have found
        best_solution = None
        old_score = None
        prev_coefs = SparsePoly(self.n_args)

        # actions to choose from
        actions = [
            i for i in range(-2, 5)
        ]

        # verbose stuff
        last_time = time.time_ns()

        best_score = 9999
        found_solutions = 0
        accepted_changes = 0
        rejected_changes = 0
        failed_changes = 0

        # do iterations
        for iter in range(max_iters):

            if verbose:
                new_time = time.time_ns()
                if iter == 0 or iter == max_iters-1 or (new_time-last_time)*1e-9 >= 0.25:
                    last_time = new_time
                    new_msg = "iteration: " + str(iter) + "/" + str(max_iters)
                    new_msg += "\nbest solution: " + ("None" if best_solution == None else str(best_solution.cost))
                    new_msg += "\nfound solutions: " + str(found_solutions)
                    new_msg += "\naccepted changes: " + str(accepted_changes)
                    new_msg += "\nrejected changes: " + str(rejected_changes)
                    new_msg += "\nfailed changes: " + str(failed_changes)
                    new_msg += "\nbest score: " + str(best_score)
                    new_msg += "\ncurrent score: " + str(old_score)
                    new_msg += "\ncurrent cost: " + str(self.circuit.cost)
                    new_msg += "\ncurrent polynomial: " + str(prev_coefs) + "\n\n"
                    sys.stdout.write(new_msg)
                    sys.stdout.flush()
                    msg = new_msg

            # try making a change to the node
            change = 0
            if len(self.circuit.nodes) > 0:
                change = random.sample(actions, 1)[0]
            old_cost = self.circuit.cost
            valid = True

            if change <= 0:
                # new root
                new_node = self.circuit.addNode(random_op(), [
                    self.circuit.root if not self.circuit.root is None else random.sample(self.circuit.all_nodes, 1)[0],
                    random.sample(self.circuit.leaf_nodes, 1)[0]
                ])
                valid = new_node.valid

            elif change == 1:
                # change op
                n = random.sample(self.circuit.nodes, 1)[0]
                valid = self.circuit.change_operation(n, not_my_op(n))

            elif change == 2:
                # insert randomly
                upper = random.sample(self.circuit.nodes, 1)[0]
                base = random.sample(upper.operands, 1)[0]
                other = random.sample(self.circuit.leaf_nodes, 1)[0]
                new_node = self.circuit.addNode(random_op(), [base, other])
                valid = new_node.valid
                if valid:
                    valid = self.circuit.change_input(upper, base, new_node)

            elif change == 3:
                # change connection
                base = random.sample(self.circuit.nodes, 1)[0]
                old_op = random.sample(base.operands, 1)[0]
                new_op = random.sample(self.circuit.all_nodes, 1)[0]
                valid = self.circuit.change_input(base, old_op, new_op)

            elif change >= 4:
                # remove rand
                # node = random.sample(self.circuit.nodes, 1)[0]
                # replace = random.sample(node.operands, 1)[0]
                # valid = self.circuit.removeNode(node, replace)
                node = random.sample(self.circuit.nodes, 1)[0]
                timeout = 0
                while sum([1 if op.is_leaf else 0 for op in node.operands]) == 0:
                    timeout += 1
                    if timeout == 10:
                        valid = False
                        break
                    node = random.sample(self.circuit.nodes, 1)[0]
                if valid:
                    replace = random.sample(node.operands, 1)[0]
                    while replace.is_leaf and sum([1 if op.is_leaf else 0 for op in node.operands]) < 2:
                        replace = random.sample(node.operands, 1)[0]
                    valid = self.circuit.removeNode(node, replace)

            self.circuit.recalc_cost()

            # make sure that this change is valid
            if valid:

                # try this circuit on the samples
                # output = self.circuit.evaluate(samples)
                # out_coefs = np.flip(np.polyfit(samples[0], output, 10).round())
                out_coefs = self.circuit.get_poly()

                if out_coefs == self.target:
                    # this is a solution
                    self.circuit.optimize()
                    found_solutions += 1
                    if best_solution is None or self.circuit.cost < best_solution.cost:
                        best_solution = self.circuit.copy()
                    if return_on_first:
                        return best_solution

                # get new score from change
                diff = self.target - out_coefs
                new_score = abs(C_cost*self.circuit.cost + C_acc*sum(abs(x)**L for x in diff.dict.values()))

                if new_score < best_score:
                    best_score = new_score

                if self.target == out_coefs or old_score == None or new_score <= old_score or random.random() <= 0.001+np.exp(-abs(new_score-old_score)/temp):
                    # take this new circuit
                    self.circuit.optimize()
                    old_score = new_score
                    accepted_changes += 1
                    prev_coefs = out_coefs
                else:
                    # undo the last change because it was bad
                    self.circuit.undo()
                    self.circuit.cost = old_cost
                    rejected_changes += 1
                self.circuit.optimize()

            else:
                failed_changes += 1

            # clean memory
            if iter % clean_freq == 0:
                self.circuit.clean()

        self.circuit.clean()
        return best_solution

def main():
    target = SparsePoly(3)
    target[1, 0, 0] = 1
    target[0, 1, 0] = 1
    target *= target
    t_1 = target.copy()
    t_2 = SparsePoly(3)
    t_2[0, 0, 1] = 2
    target *= t_2
    solution = None
    while solution == None:
        try:
            engine = SearchEngine(target, n_vals=4, costs={OPERATIONS.MULT: 1, OPERATIONS.ADD: 0.25})
            solution = engine.search(100000, 0.25, 1, 1, return_on_first=True, L=0.5)
        except KeyboardInterrupt:
            print("killed.")
            break
        except:
            pass

    print("target:", target)
    if solution == None:
        print("no solution found.")
        exit()

    outs = {}
    track_stack = []
    curr = solution.root
    ind = 1

    while True:
        if curr in track_stack:
            raise RuntimeError("Circular dependency in evaluate!")

        if curr.is_leaf:
            if curr.arg != None:
                print(ind, "--", LETTERS[curr.arg])
                if curr is solution.root:
                    break
                outs[curr] = ind
                ind += 1
                curr = track_stack.pop()
            else:
                print(ind, "--", curr.val)
                if curr is solution.root:
                    break
                outs[curr] = ind
                ind += 1
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

                    print(ind, "--", curr.operation.name, "<-", outs[curr.operands[0]], "<-",
                        outs[curr.operands[1]])
                    outs[curr] = ind
                    ind += 1
                    if curr is solution.root:
                        break

                curr = track_stack.pop()

if __name__ == '__main__':
    main()