
from sparse_poly import SparsePoly, LETTERS

import multivar_horner

import numpy as np
from dataclasses import dataclass, field
import heapq
import time
import random
import matplotlib.pyplot as plt


DECOMP_TRIALS = 10


class AnnealNode:
    def __init__(self, common, on_child, off_child, size, added_const=False):
        self.common = common.copy() if not common is None else None
        self.on_child = on_child
        self.off_child = off_child
        self.size = size
        self.added_const = added_const

    def copy(self):
        new_node = AnnealNode(self.common, None, None, self.size, added_const=self.added_const)
        if self.on_child != None:
            new_node.on_child = self.on_child.copy()
        if self.off_child != None:
            new_node.off_child = self.off_child.copy()
        return new_node

    def get_nodes(self):
        nodes = [self]
        if self.on_child is not None:
            nodes += self.on_child.get_nodes()
        if self.off_child is not None:
            nodes += self.off_child.get_nodes()
        return nodes

    def __str__(self) -> str:
        my_str = ""
        if self.on_child is not None:
            for i in range(self.common.size):
                if self.common[i] > 0:
                    my_str += "x_"+str(i)+'^'+str(round(self.common[i]))
            my_str +=  "(" + str(self.on_child) + ")"
        if self.off_child is not None:
            if len(my_str) > 0:
                my_str += " + "
            my_str += str(self.off_child)
        if self.added_const:
            if len(my_str) > 0:
                my_str += " + "
            my_str += "c"
        return my_str


def default_heuristic(m, commons):
    diff = m - np.expand_dims(np.squeeze(commons).T, 0)
    good_diff = np.min(diff, axis=1) >= 0
    good_diff = np.expand_dims(good_diff, 1)
    diff_zeros = np.full_like(m, good_diff)

    fixed = np.where(diff_zeros > 0, np.expand_dims(np.squeeze(commons).T, 0)/np.maximum(1, np.sqrt(m)), 0)
    summ = np.sum(fixed, axis=(0, 1))
    return summ


class SimpleSearch:

    def __init__(self, poly: SparsePoly, heuristic=default_heuristic, cache=True, n_rands = 0):
        self.poly = poly
        self.n = poly.n

        self.cache = cache
        self.n_rands = n_rands

        self.heuristic = heuristic

        self.cost = 0
        self.reset()        


    def reset(self, base=None):
        self.cost = 0

        cache = self.cache
        n_rands = self.n_rands

        self.commons = []
        already_used = set()

        if base is not None:
            for node in base.get_nodes():
                if not node.common is None:
                    ne = node.common.copy()
                    self.commons.append(ne)
                    already_used.add(tuple(ne))

        maxs = np.max(self._get_problem(), axis=0)
        constraints = (maxs + 1)//2

        for i in range(self.n):
            if maxs[i] > 0:
                ne = np.zeros(self.n, dtype=np.int32)
                ne[i] = 1
                if tuple(ne) not in already_used:
                    already_used.add(tuple(ne))
                    self.commons.append(ne)
        if cache:
            for i in range(self.n):
                if constraints[i] > 2:
                    for n in range(2, 1 + constraints[i]):
                        ne = np.zeros(self.n, dtype=np.int32)
                        ne[i] = n
                        if tuple(ne) not in already_used:
                            already_used.add(tuple(ne))
                            self.commons.append(ne)
        init_len = len(self.commons)
        tries = 0
        while len(self.commons) < init_len + n_rands and tries < self.n*n_rands:
            tries += 1
            ne = random.choice(self.commons).copy()
            ne[np.random.randint(0, self.n)] += 1
            if tuple(ne) not in already_used and np.all(ne <= constraints):
                already_used.add(tuple(ne))
                self.commons.append(ne)
        
        self.commons = np.stack(self.commons)
        self.commons = np.expand_dims(self.commons, 0)

    def _get_problem(self):
        problem = []
        for k in self.poly.dict.keys():
            a = np.array(k, dtype=np.int32)
            problem.append(a)

        return np.stack(problem)


    def _clean(self, m: np.ndarray, common: np.ndarray=None) -> list:

        if m.size == 0:
            raise RuntimeError("Tried to clean empty array")

        # reduce with common denominator
        if not common is None:
            self.cost += 1 # this represents a multiplication
            m[:, ] -= common

        if np.any(m < 0):
            raise RuntimeError("Negative exponent in clean.")

        cleaned = m[np.sum(m, axis=1) > 0]
        if cleaned.size < m.size:
            return cleaned, True
        return cleaned, False


    def search(self, gamma: int=None, follow_root: AnnealNode=None):

        self.reset(follow_root)
        problem = self._get_problem()

        used_commons = set()

        # stack of polynomials to be decomposed
        try:
            g, intercept = self._clean(problem)
            groups = [g]
        except:
            raise RuntimeError("Cannot clean problem. Problem is likely empty.")
        follows = [follow_root]
        
        root = AnnealNode(None, None, None, None, added_const=intercept)
        paths = [root]

        # go until we have solved every sub-polynomial
        while len(groups) > 0:

            # pop smallest remaining group
            curr_group = groups.pop()
            if curr_group.size == 0:
                raise RuntimeError("Group is empty? This shouldn't happen.")
            curr_follow = follows.pop()
            curr_path = paths.pop()

            common = None

            if not curr_follow is None:
                common = curr_follow.common

            if common is None:

                if not gamma is None and gamma == 0:
                    common = random.choice(range(self.commons.shape[1]))

                else:
                    expanded = np.full(list(curr_group.shape)+[self.commons.shape[1]], np.expand_dims(curr_group, 2))

                    scores = self.heuristic(expanded, self.commons)

                    if np.sum(scores) == 0:
                        raise RuntimeError("Group has no valid denominator choices!")
                
                    if gamma is None:
                        common = np.argmax(scores)
                    else:
                        weights = np.where(scores > 0, scores**gamma, 0)
                        weights = weights.astype(np.float64) / np.sum(weights)
                        common = np.random.choice(range(scores.size), p=weights)

                common = self.commons[0, common]

            used_commons.add(tuple(common))
            curr_path.common = common
            curr_path.size = curr_group.shape[0]

            diff = curr_group - np.expand_dims(np.squeeze(common), 0)
            good_diff = np.min(diff, axis=1) >= 0

            on_set = curr_group[good_diff, :]

            if on_set.size == 0:
                groups.append(curr_group)
                follows.append(curr_follow)
                paths.append(curr_path)
                continue

            off_set = curr_group[np.logical_not(good_diff), :]

            if off_set.size > 0:
                groups.append(off_set)
                follows.append(None if curr_follow is None else curr_follow.off_child)
                
                off_path = AnnealNode(None, None, None, None)
                curr_path.off_child = off_path
                paths.append(off_path)

            if on_set.size > 0:
                cleaned, intercept = self._clean(on_set, common)
                on_path = AnnealNode(None, None, None, None, added_const=intercept)
                curr_path.on_child = on_path
                if cleaned.size > 0:
                    groups.append(cleaned)
                    follows.append(None if curr_follow is None else curr_follow.on_child)
                    paths.append(on_path)

        for k in used_commons:
            common_costs = []

            for trial in range(DECOMP_TRIALS):
                common_costs.append(0)
                a = np.array(k, dtype=np.int32)
                while np.sum(a) > 1:
                    a[np.random.choice(range(self.n), p=np.sign(a)/np.sum(np.sign(a)))] -= 1
                    if np.min(a) < 0:
                        raise RuntimeError("invalid common factor cost calculation.")
                    common_costs[-1] += 1
                    if tuple(a) in used_commons:
                        break
                
                if common_costs[-1] <= 1:
                    break
        
            self.cost += min(common_costs)

        return self.cost, root


    def randomSearch(self, iterations: int, gamma: float=0, verbose: bool=False, save: bool=False):
        
        best_cost = None
        best_tree = None

        c_list = []
        for it in range(1, iterations+1):

            new_cost, new_tree = self.search(gamma=gamma)

            c_list.append(new_cost)

            if best_cost is None or new_cost < best_cost:
                best_cost = new_cost
                best_tree = new_tree

            if verbose:
                print("\n --- Iteration", it, "---")
                print(" - Best Cost:", best_cost)
                print(" - Prev Cost:", new_cost)

        if save:
            plt.clf()
            plt.scatter(range(len(c_list)), c_list)
            plt.xlabel("Trial")
            plt.ylabel("Cost")
            plt.title("SimpleSearch with Randomize Trials")
            plt.savefig("random_out.png")

        return best_cost, best_tree


    def annealSearch(self, iterations: int, gamma: float, temp_start: float, temp_schedule: int, base_on_greedy: bool=False, verbose: bool=False, save=False):

        prev_cost, curr_path = self.search(gamma=(None if base_on_greedy else gamma))
        best_cost = prev_cost
        best_tree = curr_path.copy()

        temp = temp_start

        acc_x = [0]
        acc_y = [prev_cost]
        rej_x = []
        rej_y = []

        def plot(a_x=acc_x, a_y=acc_y, r_x=rej_x, r_y=rej_y):
            plt.clf()
            plt.scatter(r_x, r_y, c='Red')
            plt.scatter(a_x, a_y, c='Blue')
            plt.xlabel("Trial")
            plt.ylabel("Cost")
            plt.legend(["Rejected", "Accepted"])
            plt.title("SimpleSearch with Simulated Annealing")
            plt.savefig("anneal_out.png")

        prev_time = np.floor(time.time())
        for it in range(1, 1+iterations):

            old_path = curr_path.copy()
            old_cost = prev_cost

            nodes = curr_path.get_nodes()
            if random.random() < 0.5:                
                weights = np.array([1 if nodes[i].on_child is not None else 0 for i in range(len(nodes))])
                breaker_ind = np.random.choice(range(len(nodes)), p=weights/np.sum(weights))
                breaker = nodes[breaker_ind]
                breaker.on_child = None
            else:
                weights = np.array([1 if nodes[i].off_child is not None else 0 for i in range(len(nodes))])
                breaker_ind = np.random.choice(range(len(nodes)), p=weights/np.sum(weights))
                breaker = nodes[breaker_ind]
                breaker.off_child = None
            
            new_cost, new_path = self.search(gamma=gamma, follow_root=curr_path)

            if new_cost < best_cost:
                best_cost = new_cost
                best_tree = new_path.copy()

            accepted = True
            comparer = prev_cost
            p_accept = 1 if new_cost <= prev_cost else np.exp(max(-20, (comparer - new_cost)/(temp)))
            if new_cost <= prev_cost or random.random() < p_accept:
                prev_cost = new_cost
                curr_path = new_path
                acc_x.append(it)
                acc_y.append(new_cost)
            else:
                accepted = False
                curr_path = old_path
                rej_x.append(it)
                rej_y.append(new_cost)

            if verbose:
                print("\n --- Iteration", it, "---")
                print(" - Best Cost:", best_cost)
                print(" - Prev Cost:", old_cost)
                print(" - New Cost:", new_cost)
                print(" - Temperature:", temp)
                print(" - P(Accept):", p_accept)
                print(" - Accepted:", accepted) 
            if save and np.floor(time.time()) - prev_time > it//10000:
                plot()
                prev_time = np.floor(time.time())

            if temp > 0.00001:
                temp = max(0.00001, temp - temp_start/temp_schedule)

        if save:
            plot()

        return best_cost, best_tree

    def basinSearch(self, basins: int, time_out: int, gamma: float, verbose: bool=False, save=False):

        best_cost = None
        best_tree = None

        total_seen = 0
        acc_x = []
        acc_y = []
        rej_x = []
        rej_y = []

        def plot(a_x=acc_x, a_y=acc_y, r_x=rej_x, r_y=rej_y):
            plt.clf()
            plt.scatter(r_x, r_y, c='Red')
            plt.scatter(a_x, a_y, c='Blue')
            plt.xlabel("Trial")
            plt.ylabel("Cost")
            plt.legend(["Rejected", "Accepted"])
            plt.title("BasinSearch with Simulated Annealing")
            plt.savefig("basin_out.png")

        for bas in range(basins):

            gam = (gamma if bas > 0 else None)
            prev_cost, curr_path = self.search(gamma=gam)
            total_seen += 1
            acc_x.append(total_seen)
            acc_y.append(prev_cost)

            prev_time = np.floor(time.time())
            consec_fails = 0
            while consec_fails < time_out:
                total_seen += 1

                old_path = curr_path.copy()
                old_cost = prev_cost

                nodes = curr_path.get_nodes()
                if random.random() < 0.5:                
                    weights = np.array([1 if nodes[i].on_child is not None else 0 for i in range(len(nodes))])
                    breaker_ind = np.random.choice(range(len(nodes)), p=weights/np.sum(weights))
                    breaker = nodes[breaker_ind]
                    breaker.on_child = None
                else:
                    weights = np.array([1 if nodes[i].off_child is not None else 0 for i in range(len(nodes))])
                    breaker_ind = np.random.choice(range(len(nodes)), p=weights/np.sum(weights))
                    breaker = nodes[breaker_ind]
                    breaker.off_child = None
                
                new_cost, new_path = self.search(gamma=gamma, follow_root=curr_path)

                if best_cost is None or new_cost < best_cost:
                    best_cost = new_cost
                    best_tree = new_path.copy()

                if new_cost < prev_cost:
                    consec_fails = 0
                else:
                    consec_fails += 1

                accepted = True
                if new_cost <= prev_cost:
                    prev_cost = new_cost
                    curr_path = new_path
                    acc_x.append(total_seen)
                    acc_y.append(new_cost)
                else:
                    accepted = False
                    curr_path = old_path
                    rej_x.append(total_seen)
                    rej_y.append(new_cost)

                if verbose:
                    print("\n --- Iteration", total_seen, "---")
                    print(" - Best Cost:", best_cost)
                    print(" - Prev Cost:", old_cost)
                    print(" - New Cost:", new_cost)
                    print(" - Accepted:", accepted)
                if save and np.floor(time.time()) - prev_time > total_seen//10000:
                    plot()
                    prev_time = np.floor(time.time())

        if save:
            plot()

        return best_cost, best_tree

def main():

    show_trees = True

    # generate some big random polynomial
    N = 3
    scale = 1
    coefs = 10

    target = SparsePoly(N)
    for c in range(coefs):
        k = np.round_(np.random.exponential(scale=scale, size=N)).astype(np.int32)
        target[k] = 1

    # the most basic representation just multiplies and adds every monomial one by one
    print("\nNaive Estimate:", sum([1+sum(k) for k in target.dict.keys()]))
    if show_trees: print(target)

    """ use lib to get benchmark """
    coefs = []
    keys = []
    for k in target.dict.keys():
        coefs.append(target[k])
        keys.append(k)
    horner = multivar_horner.HornerMultivarPolynomialOpt(coefs, keys, rectify_input=True, keep_tree=True)
    
    # show the benchmark
    cost, tree = horner.num_ops, horner.factorisation_tree
    print("\nHorner Computation:", horner.num_ops)
    if show_trees: print(tree)

    """ get solution using our method """
    engine = SimpleSearch(target, cache=True, n_rands=100)

    cost, tree = engine.search()
    print("\n --> Greedy Cost:", cost)
    if show_trees: print(tree)

    cost, tree = engine.randomSearch(10, gamma=None, save=True, verbose=False)
    print("\n --> Random Cost:", cost)
    if show_trees: print(tree)

    cost, tree = engine.annealSearch(100, 3, 3, 80, base_on_greedy=True, save=True, verbose=False)
    print("\n --> Anneal Cost:", cost)
    if show_trees: print(tree)

    cost, tree = engine.basinSearch(10, 500, 3, save=True)
    print("\n --> Basin Cost:", cost)
    if show_trees: print(tree)

    print("")


if __name__ == '__main__':
    main()