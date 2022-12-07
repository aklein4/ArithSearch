
from sparse_poly import SparsePoly, LETTERS

import multivar_horner

import numpy as np
import torch
from dataclasses import dataclass, field
import heapq
import time
import random
import matplotlib.pyplot as plt
import argparse

DECOMP_TRIALS = 10


DEFAULT_FLATTEN_THRESH = 100


class AnnealNode:
    def __init__(self, common, on_child, off_child, size, added_const=False):
        self.common = common.clone() if not common is None else None
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
            for i in range(self.common.numel()):
                if self.common[i] > 0:
                    my_str += "x_"+str(i)+'^'+str(round(self.common[i].item()))
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
    diff = m - commons
    good_diff = torch.min(diff, dim=1).values >= 0
    good_diff = torch.unsqueeze(good_diff, dim=1)
    good_diff.expand(-1, m.shape[1], -1)

    maxes = torch.where(m > 1, torch.sqrt(m), 1)
    fixed = torch.where(good_diff > 0, commons/maxes, 0)
    summ = torch.sum(fixed, dim=(0, 1))
    return summ


class SimpleSearch:

    def __init__(self, poly: SparsePoly, heuristic=default_heuristic, cache=True, n_rands = 0, gpu=False, flatten_thresh=DEFAULT_FLATTEN_THRESH):
        self.poly = poly
        self.n = poly.n

        self.cache = cache
        self.n_rands = n_rands

        self.heuristic = heuristic

        self.flatten_thresh = flatten_thresh

        self.device = torch.device("cuda") if gpu else torch.device("cpu")

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
                    ne = node.common.clone()
                    self.commons.append(ne)
                    already_used.add(tuple(np.round_(ne.cpu().numpy())))

        maxs = torch.max(self._get_problem(), dim=0).values
        constraints = torch.div(maxs + 1, 2, rounding_mode="floor")

        for i in range(self.n):
            if maxs[i] > 0:
                ne = torch.zeros(self.n, device=self.device)
                ne[i] = 1
                if tuple(torch.round_(ne)) not in already_used:
                    already_used.add(tuple(np.round_(ne.cpu().numpy())))
                    self.commons.append(torch.round_(ne))
        if cache:
            fixed_constraints = torch.tensor([round(constraints[i].item()) for i in range(constraints.nelement())], dtype=torch.int32)
            for i in range(self.n):
                if constraints[i] > 2:
                    for n in range(2, 1 + fixed_constraints[i]):
                        ne = torch.zeros(self.n, device=self.device)
                        ne[i] = n
                        if tuple(torch.round_(ne)) not in already_used:
                            already_used.add(tuple(np.round_(ne.cpu().numpy())))
                            self.commons.append(torch.round_(ne))
        init_len = len(self.commons)
        tries = 0
        while len(self.commons) < init_len + n_rands and tries < self.n*n_rands:
            tries += 1
            ne = random.choice(self.commons).clone()
            ne[torch.randint(self.n, size=(1,))] += 1
            if tuple(np.round_(ne.cpu().numpy())) not in already_used and torch.all(ne <= maxs):
                already_used.add(tuple(np.round_(ne.cpu().numpy())))
                self.commons.append(ne)
        
        self.commons = torch.stack(self.commons)
        self.commons = torch.transpose(self.commons, 0, 1)


    def _get_problem(self):
        problem = []
        for k in self.poly.dict.keys():
            a = torch.tensor(k, device=self.device)
            problem.append(a)

        return torch.stack(problem)


    def _clean(self, m: torch.tensor, common: torch.Tensor=None) -> list:

        if m.numel() == 0:
            raise RuntimeError("Tried to clean empty array")

        # reduce with common denominator
        if not common is None:
            self.cost += 1 # this represents a multiplication
            m[:, ] -= common

        if torch.any(m < 0):
            raise RuntimeError("Negative exponent in clean.")

        cleaned = m[torch.sum(m, dim=1) > 0]
        if cleaned.numel() < m.numel():
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
            if curr_group.numel() == 0:
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
                    expanded = torch.tile(torch.unsqueeze(curr_group, 2), (1, 1, self.commons.shape[1]))

                    scores = self.heuristic(expanded, self.commons)

                    if torch.sum(scores) == 0:
                        raise RuntimeError("Group has no valid denominator choices!")
                
                    if gamma is None:
                        common = torch.argmax(scores)
                    else:
                        weights = torch.where(scores > 0, scores**gamma, 0)
                        weights = weights / torch.sum(weights)
                        dist = torch.distributions.categorical.Categorical(weights)
                        common = dist.sample()

                common = self.commons[:, common]

            used_commons.add(tuple(np.round_(common.cpu().numpy())))
            curr_path.common = common
            curr_path.size = curr_group.shape[0]

            good_diff = torch.min(curr_group - common, dim=1).values >= 0

            on_set = curr_group[good_diff, :]

            if on_set.numel() == 0:
                groups.append(curr_group)
                follows.append(curr_follow)
                paths.append(curr_path)
                continue

            off_set = curr_group[torch.logical_not(good_diff), :]

            if off_set.numel() > 0:
                groups.append(off_set)
                follows.append(None if curr_follow is None else curr_follow.off_child)
                
                off_path = AnnealNode(None, None, None, None)
                curr_path.off_child = off_path
                paths.append(off_path)

            if on_set.numel() > 0:
                cleaned, intercept = self._clean(on_set, common)
                on_path = AnnealNode(None, None, None, None, added_const=intercept)
                curr_path.on_child = on_path
                if cleaned.numel() > 0:
                    groups.append(cleaned)
                    follows.append(None if curr_follow is None else curr_follow.on_child)
                    paths.append(on_path)

        for k in used_commons:
            common_costs = []

            for trial in range(DECOMP_TRIALS):
                common_costs.append(0)
                a = np.array(k)
                while np.sum(a) > 1:
                    a[np.random.choice(range(self.n), p=np.array(np.sign(a)/np.sum(np.sign(a))))] -= 1
                    if np.min(a) < 0:
                        raise RuntimeError("invalid common factor cost calculation.")
                    common_costs[-1] += 1
                    if tuple(np.round_(a)) in used_commons:
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


    def annealSearch(self, temp_schedule: int, gamma: float, temp_start: float=1, base_on_greedy: bool=False, verbose: bool=False, save=False):

        prev_cost, curr_path = self.search(gamma=(None if base_on_greedy else gamma))
        best_cost = prev_cost
        best_tree = curr_path.copy()

        acc_x = [0]
        acc_y = [prev_cost]
        rej_x = []
        rej_y = []

        def plot(a_x=acc_x, a_y=acc_y, r_x=rej_x, r_y=rej_y):
            plt.clf()
            plt.scatter(r_x, r_y, c='k', marker='.', s=1)
            plt.scatter(a_x, a_y, c='k', marker='x')
            plt.xlabel("Trial")
            plt.ylabel("Cost")
            plt.legend(["Rejected", "Accepted"])
            plt.title("SimpleSearch with Simulated Annealing")
            plt.savefig("anneal_out.png")

        temp = temp_start

        prev_time = np.floor(time.time())
        consec_fails = 0
        it = 0
        while it < temp_schedule or consec_fails < self.flatten_thresh:
            it += 1

            old_path = curr_path.copy()
            old_cost = prev_cost

            nodes = curr_path.get_nodes()
            if random.random() < 0.5:                
                weights = np.array([1 if nodes[i].on_child is not None else 0 for i in range(len(nodes))])
                if np.sum(weights) == 0:
                    weights += 1
                breaker_ind = np.random.choice(range(len(nodes)), p=weights/np.sum(weights))
                breaker = nodes[breaker_ind]
                breaker.on_child = None
            else:
                weights = np.array([1 if nodes[i].off_child is not None else 0 for i in range(len(nodes))])
                if np.sum(weights) == 0:
                    weights += 1
                breaker_ind = np.random.choice(range(len(nodes)), p=weights/np.sum(weights))
                breaker = nodes[breaker_ind]
                breaker.off_child = None
            
            new_cost, new_path = self.search(gamma=gamma, follow_root=curr_path)

            if new_cost < best_cost:
                best_cost = new_cost
                best_tree = new_path.copy()

            if new_cost < prev_cost:
                consec_fails = 0
            else:
                consec_fails += 1

            accepted = True
            comparer = prev_cost
            p_accept = 1 if new_cost <= prev_cost else np.exp(max(-20, (comparer - new_cost)/temp))
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


    def basinSearch(self, basins: int, gamma: float, verbose: bool=False, save=False):

        best_cost = None
        best_tree = None

        total_seen = 0
        acc_x = []
        acc_y = []
        rej_x = []
        rej_y = []

        def plot(a_x=acc_x, a_y=acc_y, r_x=rej_x, r_y=rej_y):
            plt.clf()
            plt.scatter(r_x, r_y, c='k', marker='.', s=1)
            plt.scatter(a_x, a_y, c='k', marker='x')
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
            while consec_fails < self.flatten_thresh:
                total_seen += 1

                old_path = curr_path.copy()
                old_cost = prev_cost

                nodes = curr_path.get_nodes()
                if random.random() < 0.5:                
                    weights = np.array([1 if nodes[i].on_child is not None else 0 for i in range(len(nodes))])
                    if np.sum(weights) == 0:
                        weights += 1
                    breaker_ind = np.random.choice(range(len(nodes)), p=weights/np.sum(weights))
                    breaker = nodes[breaker_ind]
                    breaker.on_child = None
                else:
                    weights = np.array([1 if nodes[i].off_child is not None else 0 for i in range(len(nodes))])
                    if np.sum(weights) == 0:
                        weights += 1
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

def main(args):

    show_trees = True

    # generate some big random polynomial
    N = 3
    scale = 3
    coefs = 25

    target = SparsePoly(N)
    for c in range(coefs):
        k = np.round_(np.random.exponential(scale=scale, size=N))
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
    engine = SimpleSearch(target, cache=True, n_rands=100, gpu=args.cuda)

    cost, tree = engine.search()
    print("\n --> Greedy Cost:", cost)
    if show_trees: print(tree)

    cost, tree = engine.randomSearch(1, gamma=None, save=True, verbose=False)
    print("\n --> Random Cost:", cost)
    if show_trees: print(tree)

    cost, tree = engine.annealSearch(1000, 3, 1, base_on_greedy=True, save=True, verbose=False)
    print("\n --> Anneal Cost:", cost)
    if show_trees: print(tree)

    cost, tree = engine.basinSearch(5, 3, save=True)
    print("\n --> Basin Cost:", cost)
    if show_trees: print(tree)

    print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python-based choicenet')

    parser.add_argument('--gpu', dest='cuda', action='store_const', const=True, default=False, 
                    help='Whether to use cuda gpu acceleration')

    args = parser.parse_args()
    main(args)