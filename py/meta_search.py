
from sparse_poly import SparsePoly, LETTERS

import multivar_horner

import numpy as np
from dataclasses import dataclass, field
import heapq
import time
import random
import matplotlib.pyplot as plt


@dataclass(order=True)
class PrioritizedGroup:
    # wrapper for set with priority
    priority: float = field(compare=True)
    l: list = field(compare=False)

def get_prioritized_group(l: list) -> PrioritizedGroup:
    # create a set that prioritizes those with smallest exponent sum first
    return PrioritizedGroup(
        sum([np.sum(k) for k in l]),
        l
    )


@dataclass(order=True)
class PrioritizedMem:
    # wrapper for array with priority
    priority: float = field(compare=True)
    k: np.ndarray = field(compare=False)

def get_prioritized_mem(k: np.ndarray):
    # get an array that prioritizes smallest sum first
    return PrioritizedMem(
        np.sum(k),
        k
    )


class MetaSearch:

    def __init__(self, poly: SparsePoly, care_about_add: bool=False, disable_mem=False, disable_cache=False, remove_early=0):
        """
        remove_early:
        0 - never remove anything early (works best on small problems)
        1 - sometimes remove based on whether a component might be reduced later (works slightly better/faster than 0 on big problems)
        2 - always remove a component if possible (worst costs but slightly faster performance)
        """
        self.poly = poly
        self.n = poly.n
    
        self.care_about_add = care_about_add
        self.disable_mem = disable_mem
        self.disable_cache = disable_cache
        self.remove_early = remove_early

        self.mem_base = []
        self.mem = []
        self.mem_set = set()
        self.reset()


    def reset(self):
        self.cost = 0
    
        self.mem_base = []
        self.mem = []
        heapq.heapify(self.mem)
        self.mem_set = set()

        self._mem_update(np.zeros(self.n, dtype=np.int32), base=True)
        for i in range(self.n):
            max_m = 2 if self.disable_mem else 1+round(np.ceil(self.poly.max_order()/2))
            for m in range(1, max_m):
                mon = np.zeros(self.n, dtype=np.int32)
                mon[i] = m
                self._mem_update(mon, base=True)
                self.cost += 1 if m > 1 else 0


    def _get_problem(self):
        problem = []
        for k in self.poly.dict.keys():
            a = np.array(k, dtype=np.int32)
            problem.append(a)

        return problem


    def _mem_update(self, monomial: np.ndarray, base: bool=False) -> None:

        k = monomial.copy()
        k.setflags(write=False)
        tup = tuple(k)
        if tup not in self.mem_set:
            if base:
                self.mem_set.add(tup)
                self.mem_base.append(get_prioritized_mem(k))
            elif not (self.disable_mem or self.disable_cache):
                self.mem_set.add(tup)
                heapq.heappush(self.mem, get_prioritized_mem(k))


    def _check_divider(self, group: list, divider, sorted=False):
        if len(group) == 0:
            return False

        if not sorted:
            group.sort(reverse=True, key=lambda x: np.sum(x))
        
        for k in group:
            if np.sum(k) <= np.sum(divider):
                break
            diff = k - divider
            if np.min(diff) >= 0 and np.sum(diff) > 0:
                return True
        
        return False


    def _clean(self, l: list, d: np.ndarray=None) -> list:
        if d is None:
            d = np.zeros(self.n, dtype=np.int32)

        if len(l) == 0:
            return []

        # if this is monomial and we are reducing it, then cache
        if len(l) == 1 and np.sum(d) > 0:
            self._mem_update(l[0])

        # reduce with common denominator
        if np.sum(d) > 0:
            self.cost += 1 # this represents a multiplication
            for k in l:
                k -= d

        # check all monomials to see if they are known
        if self.remove_early:
            l.sort(reverse=True, key=lambda x: np.sum(x))
        kept = []
        fence = False
        for k in l:
            stopping_condition = np.sum(k) == 0

            if not stopping_condition and (len(l) == 1 or self.remove_early == 2):
                stopping_condition = tuple(k) in self.mem_set

            elif not stopping_condition and self.remove_early == 1:
                stopping_condition = (
                    tuple(k) in self.mem_set and
                    not self._check_divider(kept, k, sorted=True)
                )

            if stopping_condition:
                # is known
                if np.sum(k) >= 1: # and not test:
                    # this is not a scalar, so take into account coefficient
                    self.cost += 1
                if self.care_about_add and fence:
                    # this gets added
                    fence = True # fencepost bug
                    self.cost += 1
            else:
                # not known, continue to reduce it
                kept.append(k)

        return kept


    def greedySearch(self, verbose: bool=False, max_mem: int=10000, expensive_heuristic: bool=True, meta_options: dict=None):

        self.reset()
        problem = self._get_problem()    

        annealing = False
        randomized = False
        if meta_options is not None:
            if "anneal" in meta_options:
                annealing = True
            if "random" in meta_options:
                randomized = True

        # ordered queue of polynomials to be decomposed, with smallest first
        groups = []
        heapq.heapify(groups)
        heapq.heappush(groups, get_prioritized_group(problem))

        # stuff for printing
        init_size = len(groups[0].l)
        init_time = time.time()

        my_path = []

        # go until we have solved every sub-polynomial
        loc = -1
        while len(groups) > 0:
            loc += 1

            # pop smallest remaining group, and clean without d to check if any monomials have been found since pushing
            curr_group = self._clean(heapq.heappop(groups).l)
            if len(curr_group) == 0:
                continue
            
            # print stuff
            if verbose:
                if len(curr_group) > 1:
                    num_left = sum([len(g.l) for g in groups]) + len(curr_group)
                    p_left = sum([sum([sum(elem) for elem in g.l]) for g in groups]) + sum([sum(elem) for elem in curr_group])
                    print("Remaining:", num_left, '&', p_left, "("+str(len(curr_group))+' & '+str(sum([sum(elem) for elem in curr_group]))+")", " --  Cost:", self.cost, " --  Memory Size:", len(self.mem), " --  Est. Time Left:", round(num_left * (time.time()-init_time)/(1+init_size-num_left)), "s")

            # list of common denominators that we will check
            common_list = [self.mem_base[m].k for m in range(min(len(self.mem_base), max_mem))]
            if len(common_list) < max_mem:
                common_list += [self.mem[m].k for m in range(min(len(self.mem), max_mem-len(common_list)))]

            choice = None

            if annealing and meta_options["path"] is not None and loc < len(meta_options["path"]):
                choice = meta_options["path"][loc]

            else:

                # weights of how good each common denom is
                weights = [0 for _ in range(len(common_list))]

                # iterate through all commmon denoms
                for g in range(len(common_list)):
                    if np.sum(common_list[g]) == 0:
                        # this guy sneaks in but can't be used
                        continue

                    found_sols = 0 # number of monomials that are solved completely by this denom
                    check_sol = len(self.mem_set) >= len(curr_group) # whether to check for found_sols

                    score = np.sum(common_list[g]) # cache this for speed

                    for k in curr_group:
                        # loop though all monomials in group
                        diff = k - common_list[g] # this is what is left after reduction by commons[g]
                        if np.min(diff) >= 0:
                            # if k is divisable by the commons[g] 

                            # add to weight according to heuristic
                            weights[g] +=  score if not expensive_heuristic else np.sum(common_list[g] / np.maximum(1, k)**(0.5))

                            # if reduced to solution, keep track
                            if check_sol and tuple(diff) in self.mem_set:
                                found_sols += 1

                    # if this denom gives solution for every monomial in group, then can't do better and can stop searching
                    if found_sols == len(curr_group):
                        break

                # get the best common denominator
                choice = weights.index(max(weights))

                if randomized or (annealing and meta_options["path"] is not None and loc == len(meta_options["path"])):
                    gamma = meta_options["gamma"]
                    for w in range(len(weights)):
                        weights[w] = weights[w]**gamma
                    choice = random.choices([i for i in range(len(weights))], weights=weights, k=1)[0]

            common = common_list[choice]

            if annealing:
                my_path.append((choice, len(curr_group)))

            # set that can and will be reduced
            reduce = []
            # set that cannot and will not be reduced
            keep = []
            for k in curr_group:
                diff = k - common
                if min(diff) >= 0:
                    # is divisable by common
                    reduce.append(k)
                else:
                    # is not divisible by common
                    keep.append(k)

            if len(keep) > 0:
                # if there are monomials that cannot be reduced here, they split off into their own group
                if self.care_about_add:
                    # reduce and keep are linked by an addition
                    self.cost += 1
                heapq.heappush(groups, get_prioritized_group(keep))
            
            # verbose stuff
            bef = None
            if verbose:
                bef = (sum([sum(elem) for elem in reduce]), len(reduce))

            # get the resulting group from reducing reduce with common
            after_reduce = self._clean(reduce, common)

            # verbose stuff
            if verbose:
                if len(curr_group) > 1:
                    aft = (sum([sum(elem) for elem in after_reduce]), len(after_reduce))
                    print(" -> Common:", tuple(common), " -- Reduction:", bef[1], '&', bef[0], " -- After:", aft[1], '&', aft[0])

            # if there is still more to reduce, push the remaining group to stack
            if len(after_reduce) > 0:
                heapq.heappush(groups, get_prioritized_group(after_reduce))

        if annealing:
            return self.cost, my_path
        return self.cost


    def annealSearch(self, iterations: int, gamma: float, temp_start: float, temp_schedule: int, verbose: bool=False, save=False):
    
        prev_cost = None
        best_cost = prev_cost
        curr_path = []

        acc_x = []
        acc_y = []
        rej_x = []
        rej_y = []

        temp = temp_start

        for it in range(1, iterations+1):

            new_path = []
            if curr_path != []:
                sum_list = [p[1] for p in curr_path]
                sum_list.reverse()
                for i in range(1, len(sum_list)):
                    sum_list[i] += sum_list[i-1]
                sum_list.reverse()

                delta_p = [curr_path[_][1]/(sum_list[_]+1) for _ in range(len(sum_list))]
                break_point = random.choices([i+1 for i in range(len(delta_p))], weights=delta_p, k=1)[0]

                new_path = [c[0] for c in curr_path[:break_point]]

            new_cost, new_path = self.greedySearch(meta_options={"anneal": True, "path": new_path, "gamma": gamma})

            accepted = False
            cost_delta = 0 if prev_cost is None else new_cost - prev_cost
            if cost_delta <= 0 or random.random() < np.exp(max(-20, -cost_delta/temp)):
                accepted = True
                if best_cost is None or new_cost < best_cost:
                    best_cost = new_cost
                curr_path = new_path
                prev_cost = new_cost

            if temp > 0.01:
                temp = max(0.01, temp - temp_start/temp_schedule)

            if accepted:
                acc_x.append(it)
                acc_y.append(new_cost)
            else:
                rej_x.append(it)
                rej_y.append(new_cost)

            if verbose:
                print("\n --- Iteration", it, "---")
                print(" - Best Cost:", best_cost)
                print(" - Prev Cost:", new_cost)
                print(" - Cost Delta:", cost_delta)
                print(" - Accepted:", accepted)  

        if save:
            plt.clf()
            plt.scatter(rej_x, rej_y, c='Red')
            plt.scatter(acc_x, acc_y, c='Blue')
            plt.xlabel("Trial")
            plt.ylabel("Cost")
            plt.legend(["Rejected", "Accepted"])
            plt.title("MetaSearch with Simulated Annealing")
            plt.savefig("anneal_out.png")

        return best_cost


    def randomSearch(self, iterations: int, gamma: float, verbose: bool=False, save=False):
        
        best_cost = None

        c_list = []

        for it in range(1, iterations+1):

            new_cost = self.greedySearch(meta_options={"random": True, "gamma": gamma})

            c_list.append(new_cost)

            if best_cost is None or new_cost < best_cost:
                best_cost = new_cost

            if verbose:
                print("\n --- Iteration", it, "---")
                print(" - Best Cost:", best_cost)
                print(" - Prev Cost:", new_cost)

        if save:
            if verbose:
                print("saving...")
            plt.clf()
            plt.scatter(range(len(c_list)), c_list)
            plt.xlabel("Trial")
            plt.ylabel("Cost")
            plt.title("MetaSearch with Randomize Trials")
            plt.savefig("random_out.png")
            if verbose:
                print("done")

        return best_cost


def main():

    # generate some big random polynomial
    N = 15
    scale = 3
    target = SparsePoly(N)
    more = 100000
    while more > 0:
        k = np.round_(np.random.exponential(scale=scale, size=N)).astype(np.int32)
        target[k] = 1
        more -= np.sum(k)
    print("created problem...")

    # use lib to get benchmark
    coefs = []
    keys = []
    for k in target.dict.keys():
        coefs.append(target[k])
        keys.append(k)
    horner = multivar_horner.HornerMultivarPolynomialOpt(coefs, keys, rectify_input=True, keep_tree=True)
    print("created benchmark...")

    # get solution using our method
    engine = MetaSearch(target, remove_early=1)
    t_start = time.time_ns()
    cost = engine.greedySearch(verbose=True)
    print("\n --> Cost:", cost, "("+str(round((time.time_ns() - t_start)*1e-9, 3))+" s)")

    # t_start = time.time_ns()
    # cost = engine.annealSearch(500, 1, 10, 400, verbose=False, save=True)
    # print(" --> Annealing Cost:", cost, "("+str(round((time.time_ns() - t_start)*1e-9, 3))+" s)")

    # the most basic representation just multiplies and adds every monomial one by one
    print("\nNaive Estimate:", sum([1+np.sum(np.maximum(0, np.array(k)-1)) for k in target.dict.keys()])-1)

    # show the benchmark
    print("Horner Computation:", horner.num_ops)
    og_tree = str(horner.factorisation_tree)
    fixed_tree = ""
    i = 0
    while i < len(og_tree):
        if og_tree[i] == '_':
            fixed_tree = fixed_tree[:-1]
            num = ""
            while True:
                i += 1
                try:
                    check = int(og_tree[i])
                    num += og_tree[i]
                except:
                    fixed_tree += LETTERS[int(num)]
                    break
        else:
            fixed_tree += og_tree[i]
            i += 1
    # print(fixed_tree)

    # show the polynomial we computed
    print(target)
    print("")


if __name__ == '__main__':
    main()