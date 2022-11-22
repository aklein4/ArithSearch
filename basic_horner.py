
from sparse_poly import SparsePoly

from div_search import DivSearch
from meta_search import MetaSearch

import numpy as np
from itertools import permutations


def basic_horners(poly: SparsePoly, order):

    # number of variables
    n = poly.n

    if len(order) != n:
        raise ValueError("order must have length n")

    problem = []
    for k in poly.dict.keys():
        a = np.array(k, dtype=np.int32)
        problem.append(a)

    cost = 0

    def clean(s: list, d: int):
        nonlocal cost
        if len(s) == 0:
            return []

        cost += 1 # this represents a multiplication
        for k in range(len(s)):
            s[k][d] -= 1
        # if len(s) == 1 and sum(s[0]) == 0:
        #     cost -= 1

        # check all monomials to see if they are known
        kept = []
        for k in s:
            if len(k) != 0:
                kept.append(k)

        return kept

    on_groups = [problem]
    off_groups = []

    for loc in range(n):
        var = order[loc]

        while len(on_groups) > 0:
            curr = on_groups.pop()
        
            divable = []
            no_divable = []

            for k in curr:
                if k[var] >= 1:
                    divable.append(k)
                else:
                    no_divable.append(k)

            if len(divable) > 0:
                on_groups.append(clean(divable, var))

            if len(no_divable) > 0:
                off_groups.append(no_divable)

        on_groups = off_groups
        off_groups = []

    return cost


def main():
    N = 3
    SCALE = 5
    ELEMS = 10

    target = SparsePoly(N)
    for i in range(ELEMS):
        k = np.round_(np.random.exponential(scale=SCALE, size=N)).astype(np.int32)
        target[k] = 1

    orders = list(permutations(range(N)))
    print(" --- Regular --- ")
    for ord in orders:
        print(tuple(ord), "-->", basic_horners(target, ord))

    # print("\n --- Improved --- ")
    # cost = DivSearch(target, verbose=False, test=True)
    # print(cost)

    print("\n --- Meta --- ")
    engine = MetaSearch(target, disable_mem=True)
    cost = engine.greedySearch()
    print("greedy -->", cost)
    cost = engine.annealSearch(5000, 0.25, 10, 4000, save=True)
    print("annealing -->", cost)
    cost = engine.randomSearch(1000, 0.25, save=True)
    print("random -->", cost)

    print('\n', target)

if __name__ == '__main__':
    main()