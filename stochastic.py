
import numpy as np
import random
import itertools
import sys
import time
from sklearn.metrics import mean_squared_error

from super_circuit import SuperCircuit
from circuit import OPERATIONS

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

    def __init__(self, target_func, n_args, costs={OPERATIONS.MULT: 1, OPERATIONS.ADD: 1}, n_vals=10):
        self.circuit = SuperCircuit(n_args, costs=costs)
        self.target_func = target_func
        self.n_args = n_args

        for i in range(1, n_vals+1):
            self.circuit.getValNode(i)
            self.circuit.getValNode(-i)

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

    def get_target_output(self, samples):
        return self.target_func(samples)

    def search(self, max_iters, samples_per_arg, C_cost, C_acc, p_accept, clean_freq=100, verbose=True):

        # things relating to the target
        samples = self.get_samples(samples_per_arg)
        target = self.get_target_output(samples)

        # keep track of the best solution that we have found
        best_solution = None
        old_score = None

        # actions to choose from
        actions = [
            i for i in range(6)
        ]

        # verbose stuff
        last_time = time.time_ns()

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
                    new_msg += "\ncurr cost: " + str(self.circuit.cost) + "\n\n"
                    sys.stdout.write(new_msg)
                    sys.stdout.flush()
                    msg = new_msg

            # try making a change to the node
            change = 0
            if len(self.circuit.nodes) > 0:
                change = random.sample(actions, 1)[0]
            old_cost = self.circuit.cost
            valid = True

            if change == 0:
                # new root
                self.circuit.addNode(random_op(), [
                    self.circuit.root if not self.circuit.root is None else random.sample(self.circuit.all_nodes, 1)[0],
                    random.sample(self.circuit.all_nodes, 1)[0]
                ])

            elif change == 1:
                # change op
                n = random.sample(self.circuit.nodes, 1)[0]
                valid = self.circuit.change_operation(n, not_my_op(n))

            elif change == 2:
                # insert randomly
                upper = random.sample(self.circuit.nodes, 1)[0]
                base = random.sample(upper.operands, 1)[0]
                other = random.sample(set(self.circuit.arg_leaves) | set(self.circuit.val_leaves.values()), 1)[0]
                new_node = self.circuit.addNode(random_op(), [base, other])
                valid = self.circuit.change_input(upper, base, new_node)

            elif change == 3:
                base = random.sample(self.circuit.nodes, 1)[0]
                old_op = random.sample(base.operands, 1)[0]
                new_op = random.sample(self.circuit.all_nodes, 1)[0]
                valid = self.circuit.change_input(base, old_op, new_op)

            elif change >= 4:
                # remove rand
                node = random.sample(self.circuit.nodes, 1)[0]
                replace = random.sample(node.operands, 1)[0]
                valid = self.circuit.removeNode(node, replace)

            self.circuit.recalc_cost()

            # make sure that this change is valid
            if valid:

                # try this circuit on the samples
                output = self.circuit.evaluate(samples)

                if np.array_equal(output, target):
                    # this is a solution
                    found_solutions += 1
                    if best_solution is None or self.circuit.cost < best_solution.cost:
                        best_solution = self.circuit.copy()

                # get new score from change
                new_score = C_cost*self.circuit.cost + C_acc*mean_squared_error(target, output)

                if np.array_equal(output, target) or old_score == None or new_score <= old_score or random.random() <= p_accept:
                    # take this new circuit
                    old_score = new_score
                    accepted_changes += 1
                else:
                    # undo the last change because it was bad
                    self.circuit.undo()
                    self.circuit.cost = old_cost
                    rejected_changes += 1
            
            else:
                failed_changes += 1

            # clean memory
            if iter % clean_freq == 0:
                self.circuit.clean()

        self.circuit.clean()
        return best_solution

def main():
    engine = SearchEngine(lambda x: 3*x[0]**4 + 4*x[0]**2 + 2*x[0] + 1, 1, n_vals=2, costs={OPERATIONS.MULT: 1, OPERATIONS.ADD: .25})
    engine.search(100000, 10, 1, 1, 0.5)

if __name__ == '__main__':
    main()