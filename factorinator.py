
from sparse_poly import SparsePoly, LETTERS, poly_MSE
from circuit import OpInfo, OPERATIONS

import multivar_horner

import numpy as np
import random


class Nom:
    def __init__(self, a: np.ndarray):
        self.a = a
    def __hash__(self):
        return hash(str(self.a))
    def __eq__(self, other):
        return np.array_equal(self.a, other.a)


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
    return np.sum(np.ceil(np.log2(1+np.abs(k))))


def generalized_horners(poly, verbose:bool=False, mem=None, sampling=None, early_stopping=None, care_about_add=True):
    if mem is None:
        mem = set()

    group = set()
    if isinstance(poly, SparsePoly):
        for k in poly.dict.keys():
            a = np.array(k)
            group.add(Nom(a))
    else:
        for k in poly:
            group.add(k)

    if len(group) == 0:
        return 0

    n = list(group)[0].a.size

    def get_clean(s: set, meem=mem, caar=care_about_add):
        clean = set()
        ms = 0
        for k in s:
            if np.sum(k.a) > 1 and k not in meem:
                clean.add(k)
            else:
                if np.sum(k.a) >= 1:
                    ms += 1
                if caar:
                    ms += 1
        return clean, ms

    def get_reduced(s: set, d: np.ndarray):
        new_group = set()
        for k in s:
            l = k.a - d
            if min(l) < 0:
                raise RuntimeError("Negative exponent in reduction.")
            new_group.add(Nom(l))

        return new_group

    group, cost = get_clean(group)

    if len(group) == 0:
        return max(cost - 1, 0) if care_about_add else cost # fencepost of addition

    # k_list = list(group)
    # k_list.sort(reverse=True, key=lambda k: np.sum(k))


    # commons = []
    # for k in k_list:

    #     found = False
    #     for g in range(len(commons)):
    #         new_d = tuple([min(commons[g][_], k[_]) for _ in range(len(k))])
    #         if sum(new_d) > 0:
    #             commons[g] = new_d
    #             found = True
    #             break

    #     if not found:
    #         commons.append(k)

    constraints = np.amax(np.stack([nom.a for nom in group]), axis=0)

    commons = []
    for i in range(n):
        for m in range(1, int(constraints[i])+1):
            check = np.zeros(n)
            check[i] = m
            if np.min(check - constraints) <= 0:
                commons.append(check)
    for m in mem:
        if np.min(m.a - constraints) <= 0:
            commons.append(m.a)

    if len(commons) == 0:
        raise RuntimeError("Empty constrained common denominator groups.")

    groups = None
    best_groups_cost = None
    master_group = group.copy()
    for iter in range(1 if sampling is None else 5):
        group = master_group.copy()
        trial_groups = [set() for _ in range(len(commons))]
        while len(group) > 0:

            weights = [0 for _ in range(len(commons))]
            perfects = [0 for _ in range(len(commons))]
            for g in range(len(weights)):
                score = np.log2(float(1+np.sum(commons[g])))
                found = 0
                for k in group:
                    diff = k.a - commons[g]
                    if np.min(diff) >= 0:
                        weights[g] += score
                        found += 1
                if found == len(group):
                    perfects[g] = weights[g]

            if sum(perfects) > 0:
                weights = perfects

            highest = weights.index(max(weights))
            if sum(perfects) == 0 and not sampling is None:
                highest = random.choices([i for i in range(len(weights))], weights=[w**sampling for w in weights], k=1)[0]

            for k in group.copy():
                diff = k.a - commons[highest]
                if min(diff) >= 0:
                    trial_groups[highest].add(k)
                    group.remove(k)
    
        overall_cost = 0
        if not sampling is None:
            for gro in trial_groups:
                if len(gro) > 0:
                    overall_cost += np.sum(np.var(np.stack([k.a for k in gro]), axis=0))

        if groups is None or overall_cost < best_groups_cost:
            groups = trial_groups
            best_groups_cost = overall_cost

    order = [i for i in range(len(groups))]
    order.sort(reverse=True, key=lambda x: len(groups[x]))
    for g in order:

        curr_group = groups[g]
        curr_d = commons[g]

        if len(curr_group) == 0:
            continue

        # verbose info
        if len(curr_group) > 1:
            if verbose:
                # print(commons)
                print(tuple(curr_d))

        # save every length 1 group that we find
        savers = set()
        if len(curr_group) == 1:
            savers.add(list(curr_group)[0])

        cost += 1
        curr_group = get_reduced(curr_group, curr_d)
        if len(curr_group) == 1:
            savers.add(list(curr_group)[0])

        curr_group, d_cost = get_clean(curr_group)
        cost += d_cost
        if len(curr_group) == 1:
            savers.add(list(curr_group)[0])

        if len(curr_group) != 0:
            cost += generalized_horners(curr_group, verbose=verbose, mem=mem, sampling=sampling, early_stopping=(None if early_stopping is None else early_stopping-cost), care_about_add=care_about_add)

        for m in savers:
            mem.add(m)

        if np.sum(curr_d) > 1 and Nom(curr_d) not in mem:
            cost += generalized_horners(set([Nom(curr_d)]), verbose=verbose, mem=mem, sampling=sampling, early_stopping=(None if early_stopping is None else early_stopping-cost), care_about_add=care_about_add)

        if not early_stopping is None and cost >= early_stopping:
            return early_stopping + 1

    return cost


def stochastic_horners(poly, iters, gamma, verbose=False, care_about_add=True):

    non_stoch_sol = generalized_horners(poly, care_about_add=care_about_add)
    best_solution = non_stoch_sol

    for iter in range(iters):
        sol = generalized_horners(poly, sampling=gamma, care_about_add=care_about_add)
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

    target = SparsePoly(10)
    more = 100
    for m in range(more):
        k = tuple([random.randrange(10) for i in range(10)])
        target.dict[k] = 1
        more -= sum(k)
    print("running...")

    coefs = []
    keys = []
    for k in target.dict.keys():
        coefs.append(target[k])
        keys.append(k)
    horner = multivar_horner.HornerMultivarPolynomialOpt(coefs, keys, rectify_input=True, keep_tree=True)
    print("got horner...\n")

    my_mem = set()
    cost = generalized_horners(target, verbose=False, care_about_add=False, mem=my_mem)
    print([tuple(m.a) for m in my_mem])

    # print(target, '\n')
    # print("Multiplication Focussed:")
    print(" --> Cost:", cost)
    print("")
    # print("Additions (approximate):", sum(target.dict.values()), '\n')

    # print("Generalized:")
    # print("Multiplications (approximate):", cost+len(target.dict.values()))
    # print("Additions (approximate):", len(target.dict.values()), '\n')

    print("Naive Estimate:", sum([log_cost(k) for k in target.dict.keys()]))

    claim = horner.num_ops
    tree = str(horner.factorisation_tree)
    print("Horner Computation:", claim)
    # print(tree)
    print("")

    #print(target)

    exit()

    cum_cost = {"cost": 0}
    solutions = search(target, 5000, cum_cost)
    print("COST:", cum_cost["cost"])
    # for sol in solutions:
    #     print(sol)

if __name__ == '__main__':
    main()