
from sparse_poly import SparsePoly, LETTERS, poly_MSE
from circuit import OpInfo, OPERATIONS

import multivar_horner

import numpy as np
from dataclasses import dataclass, field
import heapq
import random
import time

def log_cost(k):
    return np.sum(np.ceil(np.log2(1+np.abs(k))))


@dataclass(order=True)
class PrioritizedGroup:
    priority: float = field(compare=True)
    s: set = field(compare=False)

def get_prioritized_group(s: set):
    return PrioritizedGroup(
        sum([np.sum(k.pub) for k in s]),
        s
    )

@dataclass(order=True)
class PrioritizedMem:
    priority: float = field(compare=True)
    s: np.ndarray = field(compare=False)

def get_prioritized_mem(k: np.ndarray):
    return PrioritizedMem(
        np.sum(k),
        k
    )

class Nom:
    def __init__(self, a: np.ndarray):
        self.pub = a.copy()
        self.priv = a.copy()
        self.target = np.zeros_like(a, dtype=np.int32)
        self.target_score = 0
        self.rej = False

    def check(self, pot: np.ndarray):
        diff = self.priv - pot
        if np.min(diff) < 0:
            return
        if sum(pot) > self.target_score:
            self.pub = self.priv - pot
            self.target = pot.copy()
            self.target_score = sum(pot)

    def copy(self):
        n = Nom(self.priv)
        n.priv = self.priv.copy()
        n.pub = self.pub.copy()
        n.target = self.target.copy()
        n.target_score=  self.target_score

    def peek_change(self, pot: np.ndarray):
        diff = self.priv - pot
        if np.min(diff) < 0:
            return None
        return sum(pot)

    def valid_target(self):
        diff = self.priv - self.target
        if np.min(diff) < 0:
            self.target = np.zeros_like(self.target)
            self.target_score = 0
            self.pub = self.priv.copy()
            return False
        return True

    def sub(self, k: np.ndarray):
        self.pub -= k
        self.priv -= k

    def __hash__(self):
        return hash(str(self.priv))
    def __eq__(self, other):
        return np.array_equal(self.priv, other.priv)


def singlet(s: set):
    return list(s)[0]


def caching_horners(poly: SparsePoly, verbose:bool=False, care_about_add=True, max_mem=200):
    n = poly.n

    problem = set()
    for k in poly.dict.keys():
        a = np.array(k, dtype=np.int32)
        problem.add(Nom(a))

    cost = 0

    groups = []
    heapq.heapify(groups)
    heapq.heappush(groups, get_prioritized_group(problem))

    mem = [get_prioritized_mem(np.zeros(n, dtype=np.int32))]
    heapq.heapify(mem)
    mem_str = set([tuple(np.zeros(n, dtype=np.int32))])

    def mem_update(nommer: Nom):
        nonlocal groups
        nonlocal mem
        k = nommer.priv.copy()

        if tuple(k) not in mem_str:
            # for group in groups:
            #     for p in group.s:
            #         p.check(k)
            heapq.heappush(mem, get_prioritized_mem(k))
            mem_str.add(tuple(k))

    for i in range(n):
        for m in range(1, 1+round(np.ceil(poly.max_order()/2))):
            mon = np.zeros(n, dtype=np.int32)
            mon[i] = m
            mem_update(Nom(mon))
            cost += 1 if m > 1 else 0

    def clean(s: set, d: np.ndarray=np.zeros(n, dtype=np.int32)):
        nonlocal cost
        nonlocal care_about_add

        if len(s) == 0:
            return set()
        if len(s) == 1 and np.sum(d) > 0:
            mem_update(singlet(s))

        if np.sum(d) > 0:
            cost += 1
        for k in s:
            k.sub(d)
        # for k in s:
        #     if not k.valid_target():
        #         for m in mem:
        #             k.check(m)

        cache = None
        if len(s) == 1:
            cache = singlet(s)

        kept = set()
        fence = False
        for k in s.copy():
            if tuple(k.priv) in mem_str or sum(k.priv) <= 1:
                if np.sum(k.priv) >= 1:
                    cost += 1
                if care_about_add and fence:
                    fence = True
                    cost += 1
            else:
                kept.add(k)
    
        if len(kept) == 0 and cache is not None:
            mem_update(cache)

        return kept

    init_size = len(groups[0].s)
    init_time = time.time()
    while len(groups) > 0:
        curr_group = clean(heapq.heappop(groups).s)
        if len(curr_group) == 0:
            continue

        if verbose:
            if len(curr_group) > 1:
                num_left = sum([len(g.s) for g in groups]) + len(curr_group)
                print("Nomials Remaining:", num_left, "("+str(len(curr_group))+")", " --  Cost:", cost, " --  Memory Size:", len(mem), " --  Est. Time Left:", round(num_left * (time.time()-init_time)/(1+init_size-num_left)), "s")
                # print("\n----\n")
                # print([tuple(k.pub) for k in curr_group], '\n')
                # print([tuple(k.priv) for k in curr_group], '\n')

        common_list = [mem[m].s for m in range(min(len(mem), max_mem))]

        weights = [0 for _ in range(len(common_list))]
        # perfects = [0 for _ in range(len(common_list))]
        for g in range(len(weights)):
            if np.sum(common_list[g]) == 0:
                continue

            found_sols = 0
            score = np.sum(common_list[g])
            check_sol = len(mem) >= len(curr_group)
            # found = 0
            for k in curr_group:
                diff = k.priv - common_list[g]
                if np.min(diff) >= 0:
                    weights[g] += score

                    if check_sol and tuple(diff) in mem_str:
                        found_sols += 1

            if found_sols == len(curr_group):
                break

            #         found += 1
            # if found == len(curr_group):
            #     perfects[g] = weights[g]

        # if sum(perfects) > 0:
        #     weights = perfects

        common = common_list[weights.index(max(weights))]

        reduce = set()
        keep = set()
        for k in curr_group:
            diff = k.priv - common
            if min(diff) >= 0:
                reduce.add(k)
            else:
                keep.add(k)

        if len(keep) > 0:
            if care_about_add:
                cost += 1
            heapq.heappush(groups, get_prioritized_group(keep))
        
        after_reduce = clean(reduce, common)

        if len(after_reduce) > 0:
            heapq.heappush(groups, get_prioritized_group(after_reduce))

    return cost


def main():
    # target = SparsePoly(3)
    # target[1, 0, 0] = 1
    # target[0, 1, 0] = 1
    # target *= target
    # t_2 = SparsePoly(3)
    # t_2[0, 0, 1] = 2
    # target *= t_2
    # target += t_2*2 + 2
    # target *= target
    # target[4, 5, 3] = 4
    # target[3, 3, 2] = 7
    # target *= target
    # target[17, 13, 14] = 1
    # target *= target

    target = SparsePoly(5)
    more = 100000
    while more > 0:
        k = tuple([random.randrange(20) for i in range(5)])
        target.dict[k] = 1
        more -= sum(k)
    print("running...")

    coefs = []
    keys = []
    for k in target.dict.keys():
        coefs.append(target[k])
        keys.append(k)
    horner = multivar_horner.HornerMultivarPolynomial(coefs, keys, rectify_input=True, keep_tree=True)
    print("got horner...\n")

    print(horner.num_ops)

    cost = caching_horners(target, verbose=True, care_about_add=False)

    print(" --> Cost:", cost)
    print("")

    print("Naive Estimate:", sum([1+np.sum(np.maximum(0, np.array(k)-1)) for k in target.dict.keys()]))

    claim = horner.num_ops
    tree = str(horner.factorisation_tree)
    # for i in range(len(tree)):
    #     if tree[i] == '^':
    #         claim -= int(tree[i+1])-1
    print("Horner Computation:", claim)
    # print(horner.factorisation_tree)

    print(target)

    print("")


if __name__ == '__main__':
    main()