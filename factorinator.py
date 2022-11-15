
from sparse_poly import SparsePoly, LETTERS, poly_MSE
from circuit import OpInfo, OPERATIONS

import multivar_horner

import numpy as np
import random


COST_SAVE_ITERS = 10


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

def log_cost(k):
    return sum([np.ceil(np.log2(float(1+abs(k[i])))) for i in range(len(k))])


def generalized_horners(poly, verbose:bool=False, mem=None, sampling=None, early_stopping=None):
    if mem is None:
        mem = set()

    group = poly
    if isinstance(poly, SparsePoly):
        group = set(poly.dict.keys())

    def get_clean(s: set, meem=mem):
        clean = set()
        for k in s:
            if sum(k) > 1 and k not in meem:
                clean.add(k)
        return clean

    def get_reduced(s: set, d: tuple):
        new_group = set()
        for k in s:
            l = tuple([k[_] - d[_] for _ in range(len(k))])
            if min(l) < 0:
                raise RuntimeError("Negative exponent in reduction.")
            new_group.add(l)

        return new_group

    group = get_clean(group)

    if len(group) == 0:
        return 0

    k_list = list(group)
    k_list.sort(reverse=True, key=lambda k: sum(k))

    commons = list()
    for k in k_list:

        found = False
        for g in range(len(commons)):
            new_d = tuple([min(commons[g][_], k[_]) for _ in range(len(k))])
            if sum(new_d) > 0:
                commons[g] = new_d
                found = True
                break

        if not found:
            commons.append(k)

    if not sampling is None:
    #     # for i in range(len(k_list[0])):
    #     #     check = [0 for _ in range(len(k_list[0]))]
    #     #     check[i] = 1
    #     #     check = tuple(check)
    #     #     if check not in commons:
    #     #         commons.append(check)
        commons += list(mem)


    groups = [set() for _ in range(len(commons))]
    for k in k_list:
        best = None
        best_score = None
        sample_inds = []
        sample_weights = []

        for g in range(len(commons)):
            diff = tuple([(k[_]-commons[g][_]) for _ in range(len(k))])
            if min(diff) < 0:
                continue
            sq_diff = tuple([diff[_]**2 for _ in range(len(diff))])
            score = sum(sq_diff)
            if not sampling is None:
                sample_inds.append(g)
                sample_weights.append(1/(.1+score)**sampling)
            if best == None or score < best_score:
                best = g
                best_score = score

        if best == None:
            raise RuntimeError("Unfound common denominator: " + str(k) + " not in " + str(commons))
        else:
            if not sampling is None:
                groups[random.choices(sample_inds, weights=sample_weights, k=1)[0]].add(k)
            else:
                groups[best].add(k)
    
    cost = 0
    order = [i for i in range(len(groups))]
    order.sort(reverse=True, key=lambda x: len(groups[x]))
    for g in order:

        curr_group = groups[g]
        curr_d = commons[g]

        if len(curr_group) == 0:
            continue

        needed = curr_d
        if len(curr_group) > 1:
            # if verbose:
            #     print(commons)
            #     print(curr_group, "-->", curr_d, '\n')

            cost += 1
            
            curr_group = get_reduced(curr_group, curr_d)
            cost += generalized_horners(curr_group, verbose=verbose, mem=mem, sampling=sampling, early_stopping=(None if early_stopping is None else early_stopping-cost))
        
        else:
            needed = list(curr_group)[0]

        if needed not in mem:
            for trial in range(1):
                best_diff = log_cost(needed)
                best_tup = needed
                for m in mem:
                    diff = tuple([(needed[_]-m[_]) for _ in range(len(m))])
                    if min(diff) < 0:
                        continue
                    if 1+log_cost(diff) < best_diff:
                        best_diff = 1+log_cost(diff)
                        best_tup = m
                cost += best_diff
                needed = tuple([needed[i] - best_tup[i] for i in range(len(needed))])
                mem.add(needed)
                if sum(needed) == 0:
                    break

        if not early_stopping is None and cost >= early_stopping:
            return early_stopping + 1

    print(cost)
    return cost


def stochastic_horners(poly, iters, gamma, verbose=False):

    non_stoch_sol = generalized_horners(poly)
    best_solution = non_stoch_sol

    for iter in range(iters):
        sol = generalized_horners(poly, sampling=gamma, early_stopping=best_solution)
        if sol < best_solution:
            best_solution = sol
        if verbose:
            print("\nFound Solution:", sol)
            print("Best Solution:", best_solution)
    
    if verbose:
        print("\nGreedy Solution:", non_stoch_sol)
        print("Best Solution:", best_solution)
        print("")

    return best_solution


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

    target.clean()

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
    c = SparsePoly(n)
    for k in target.dict.keys():
        if target[k] != 0:
            if k not in check.dict.keys() or check[k] == 0:
                c[k] = 1

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
        delta[k] = -1 if k in ab[changer].dict.keys() and ab[changer][k] > 0 else 1
            
        if random.random() < 0.1:
            delta -= ab[changer]

        ab[changer] += delta
        # print(ab[0], "&", ab[1])

        ab[changer].clean()

        valid = True
        check = (ab[0]*ab[1])
        check.clean()
        for k in check.dict.keys():
            if k not in target.dict.keys():
                ab[changer] -= delta
                ab[changer].clean()
                valid = False
                break
        if not valid:
            continue


        c = SparsePoly(n)
        for k in target.dict.keys():
            if target[k] != 0:
                if k not in check.dict.keys() or check[k] == 0:
                    c[k] = 1

        new_cost = total_cost([ab[0], ab[1], c])

        # print(prev_cost, "-->", new_cost)

        valid = len(ab[changer]) > 0
        # print(valid)

        if valid and new_cost < best_cost:
            best = [ab[0].copy(), ab[1].copy(), c.copy()]
            best_cost = new_cost

        if valid and (new_cost < prev_cost or random.random() < 0.01):
            prev_cost = new_cost
        else:
            ab[changer] -= delta
            ab[changer].clean()

        # print(ab[0], "&" ,ab[1])
        # input()

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
    target[4, 5, 3] = 4
    target[3, 3, 2] = 7
    target *= target
    target[17, 13, 14] = 1
    target *= target

    target = SparsePoly(5)
    to_go = 1000000
    while to_go > 0:
        k = tuple([random.randrange(10) for i in range(5)])
        target.dict[k] = 1
        to_go -= sum(k)
    print("running...")

    coefs = []
    keys = []
    for k in target.dict.keys():
        coefs.append(target[k])
        keys.append(k)
    horner = multivar_horner.HornerMultivarPolynomial(coefs, keys, rectify_input=True, keep_tree=True)
    print("got horner...")

    cost = generalized_horners(target, verbose=True)

    # print(target, '\n')
    print("Multiplication Focussed:")
    print("Multiplications:", cost)
    print("Additions (approximate):", sum(target.dict.values()), '\n')

    print("Generalized:")
    print("Multiplications (approximate):", cost+len(target.dict.values()))
    print("Additions (approximate):", len(target.dict.values()), '\n')

    print("Naive Estimate:", sum([log_cost(k) for k in target.dict.keys()]))
    print("")

    print("Horner Computation:", horner.num_ops)
    print("")

    exit()

    cum_cost = {"cost": 0}
    solutions = search(target, 5000, cum_cost)
    print("COST:", cum_cost["cost"])
    # for sol in solutions:
    #     print(sol)

if __name__ == '__main__':
    main()