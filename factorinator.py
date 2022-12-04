
from sparse_poly import SparsePoly, LETTERS, poly_MSE
from circuit import OpInfo, OPERATIONS

import multivar_horner

import numpy as np
import random


def make_clean(poly: SparsePoly):
    clean = poly.copy()

    check = tuple([0 for _ in range(clean.n)]) 
    if check in clean.dict.keys():
        clean.dict.pop(check)

    return clean


def log_cost(k):
    return np.sum(np.ceil(np.log2(1+np.abs(k))))


def set_to_one(poly: SparsePoly):
    poly.dict = {}
    poly.dict[tuple([0 for _ in range(poly.n)])] = 1


def search(target: SparsePoly, per_iters: int) -> int:
    
    groups = []
    total_cost = 0

    groups.append(make_clean(target))

    while len(groups) > 0:

        group = groups.pop()

        best_trio = None
        best_cost = None

        for it in range(per_iters):

            g_set = set(group.dict.keys())

            ab = [[], []]
            ab_sets = [set(), set()]

            first_k = np.random.randint(0, group.max_order()+1)
            first = np.zeros(group.n, dtype=np.int64)
            first[random.randrange(group.n)] = first_k
            for g in group.dict.keys():
                diff = np.array(g) - first
                if np.min(diff) >= 0:
                    ab_sets[0].add(tuple(diff))
            while len(ab_sets[0]) <= 1 or sum(first) == 0:
                first = np.random.randint(0, group.max_order()+1, size=(group.n,))
                ab_sets[0] = set()
                for g in group.dict.keys():
                    diff = np.array(g) - first
                    if np.min(diff) >= 0:
                        ab_sets[0].add(tuple(diff))
            ab[0].append(tuple(first))

            second = random.choice(list(ab_sets[0]))
            ab_sets[0].remove(second)
            ab[1].append(second)
            for g in group.dict.keys():
                diff = np.array(g) - np.array(second)
                if np.min(diff) >= 0 and tuple(diff) not in ab[0]:
                    ab_sets[1].add(tuple(diff))

            g_set.remove(tuple([ab[0][0][_]+ab[1][0][_] for _ in range(group.n)]))

            if len(ab_sets[0]) == 0 or len(ab_sets[1]) == 0:
                continue

            can_grow = [0, 1]
            while len(can_grow) > 0:

                on = random.choice(can_grow)
                off = 1 - on

                choice = random.choice(list(ab_sets[off]))
                ab_sets[off].remove(choice)

                ab[on].append(choice)
                arr = np.array(choice)
                new_set = set()
                for g in group.dict.keys():
                    diff = np.array(g) - arr
                    if np.min(diff) >= 0:
                        new_set.add(tuple(diff))

                ab_sets[off] = ab_sets[off].intersection(new_set)
                if len(ab_sets[off]) == 0:
                    can_grow.remove(on)

            a = SparsePoly(group.n)
            for k in ab[0]:
                a[k] = 1
            b = SparsePoly(group.n)
            for k in ab[1]:
                b[k] = 1

            c = group - (a*b)

            cost = sum(sum(k) for k in a.dict.keys())**2 + sum(sum(k) for k in b.dict.keys())**2 + sum(sum(k) for k in c.dict.keys())**2
            if min(c.dict.values()) >= 0 and (best_cost is None or cost < best_cost):
                best_trio = [a, b, c]
                best_cost = cost
        
        for t in best_trio:
            print(t, '\n')
        exit()
            
                

def main():
    target = SparsePoly(3)
    target[1, 0, 0] = 1
    target[0, 1, 0] = 1
    target[1, 0, 1] = 1
    target[0, 0, 2] = 1
    target *= target
    t_2 = SparsePoly(3)
    t_2[0, 0, 1] = 1
    t_2[1, 0, 1] = 1
    t_2[0, 1, 0] = 1
    print(target)
    print(t_2)
    print("")
    target *= t_2
    target *= target
    print(target, '\n')

    search(target, 1000)

    # target = SparsePoly(3)
    # t_2 = SparsePoly(3)
    # target[1, 0, 1] = 1
    # target[1, 1, 0] = 1

    # t_2[2, 0, 0] = 1
    # t_2[0, 1, 2] = 1

    # print(target*t_2)

if __name__ == '__main__':
    main()