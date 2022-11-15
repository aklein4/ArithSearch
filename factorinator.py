
from sparse_poly import SparsePoly, LETTERS, poly_MSE
from circuit import OpInfo, OPERATIONS

import numpy as np
import random


def make_clean(poly: SparsePoly, key_lib=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]):
    outliers = SparsePoly(poly.n)
    clean = SparsePoly(poly.n)

    for k in poly.dict.keys():
        if poly.dict[k] == 0:
            continue
        if k in key_lib:
            outliers[k] = poly[k]
        else:
            clean[k] = poly[k]

    if len(clean) <= 1:
        return clean, tuple([0 for _ in range(poly.n)]), outliers

    common_d = tuple([min([k[i] for k in clean.dict.keys()]) if len(clean.dict) > 0 else 0 for i in range(clean.n)])

    final = SparsePoly(poly.n)

    for k in clean.dict.keys():
        adjusted = tuple([k[i] - common_d[i] for i in range(len(common_d))])
        final[adjusted] = clean[k]

    return final, common_d, outliers


def get_cost(p: SparsePoly):

    cleaned = make_clean(p, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])[0]

    cost = 0

    for k in cleaned.dict.keys():
        if cleaned.dict[k] != 0:
            cost += sum(k)
            cost += np.log(float(abs(cleaned.dict[k])))

    return cost


def total_cost(polys):
    cost = 0
    for poly in polys:
        cost += get_cost(poly)**2
    
    return cost


def search(target, iters, cum_cost):
    dirty_target = target.copy()

    for k in dirty_target.dict.keys():
        if target[k] != 0:
            target[k] = 1

    print(dirty_target)

    target, t_d, t_out = make_clean(target)
    if max(t_d) > 0:
        cum_cost["cost"] += 1

    check = make_clean(target)[0]
    if len(check) <= 1:
        # if sum([1 if abs(v) > 0 else 0 for v in target.dict.values()]) == 1:
        #     cum_cost["cost"] += sum([0 if target[k]==0 else sum(k) for k in target.dict.keys()])
        print("--- --- ---")
        return [dirty_target]

    print("vvv vvv vvv ")

    n = target.n
    max_order = [max([k[i] for k in target.dict.keys()]) if len(target) > 0 else 0 for i in range(n)]

    ab = [None, None]
    ab[0] = SparsePoly(n)
    ab[0][[random.randint(0, 1+max_order[_]//2) for _ in range(n)]] = 1
    ab[1] = SparsePoly(n)
    ab[1][[random.randint(0, 1+max_order[_]//2) for _ in range(n)]] = 1
    check = (ab[0]*ab[1])
    for k in check.dict.keys():
        if check[k] != 0:
            check[k] = 1
    c = target - check

    best = [ab[0].copy(), ab[1].copy(), c.copy()]
    prev_cost = total_cost([ab[0], ab[1], c])
    best_cost = prev_cost
    k = tuple([random.randint(0, 1+max_order[_]//2) for _ in range(n)])
    momentum = 0
    rejected = False

    for iter in range(iters):
        changer = random.randrange(2)

        delta = SparsePoly(n)
        k = tuple([random.randint(0, 1+max_order[_]//2) for _ in range(n)])
        delta[k] = random.choice([-1, 1])
            
        if random.random() < 0.01:
            delta -= ab[changer]
        ab[changer] += delta

        check = (ab[0]*ab[1])
        for k in check.dict.keys():
            if check[k] != 0:
                check[k] = 1
        c = target - check

        new_cost = total_cost([ab[0], ab[1], c])

        not_zero = sum([abs(val) for val in ab[changer].dict.values()]) > 0

        if not_zero and new_cost < best_cost:
            best = [ab[0].copy(), ab[1].copy(), c.copy()]
            best_cost = new_cost

        if new_cost > prev_cost:
            rejected = True
            momentum = 0
        else:
            rejected = False

        if not_zero and (new_cost < prev_cost or random.random() < 0.01):
            prev_cost = new_cost
        else:
            ab[changer] -= delta

    # print([str(b) for b in best])
    cum_cost["cost"] += 1
    return (
        search(best[0], iters, cum_cost) +
        search(best[1], iters, cum_cost) +
        search(best[2], iters, cum_cost)
    )


def main():
    target = SparsePoly(3)
    target[1, 0, 0] = 1
    target[0, 1, 0] = 1
    target *= target
    t_2 = SparsePoly(3)
    t_2[0, 0, 1] = 2
    target *= t_2
    target += t_2*2 + 2
    target *= target
    # target[4, 5, 3] = 4
    # target[3, 3, 2] = 7
    # target *= target
    
    cum_cost = {"cost": 0}
    solutions = search(target, 1000, cum_cost)
    print("COST:", cum_cost["cost"])
    # for sol in solutions:
    #     print(sol)

if __name__ == '__main__':
    main()