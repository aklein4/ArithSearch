
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


def get_open(p, off):

    this_open = None

    for d in off.dict.keys():
        new_set = SparsePoly(p.n)

        for k in p.dict.keys():
            diff = np.array(k) - np.array(d)

            if np.min(diff) >= 0 and p[k] - off[d] >= 0:
                new_set[tuple(diff)] = p[k] // off[d]
        
        if this_open is None:
            this_open = new_set

        else:
            for k in this_open.dict.keys():

                if k not in new_set.dict.keys():
                    this_open[k] = 0
                
                else:
                    this_open[k] = min(this_open[k], new_set[k])

            this_open.clean()

    return this_open


def search(target: SparsePoly, per_iters: int) -> int:
    n = target.n

    groups = []

    groups.append(make_clean(target))

    while len(groups) > 0:

        group = groups.pop()

        best_combo = None
        best_cost = None

        for it in range(per_iters):
            print(it)

            ab = [SparsePoly(n), SparsePoly(n)]
            
            a_start = [0 for _ in range(n)]
            a_start[random.randrange(n)] = 1
            ab[0][a_start] = 1

            b_set = get_open(target, ab[0])
            b_choice = random.choice(list(b_set.dict.keys()))
            ab[1][b_choice] = 1

            open_sets = [get_open(target - ab[0]*ab[1], ab[1]), get_open(target - ab[0]*ab[1], ab[0])]
            for i in range(len(open_sets)):
                for d in ab[i].dict.keys():
                    if d in open_sets[i].dict.keys():
                        del open_sets[i].dict[d]

            while True:

                can_add = []
                for i in range(len(open_sets)):
                    if len(open_sets[i]) > 0:
                        can_add.append(i)

                if len(can_add) == 0:
                    break

                add_choice = random.choice(can_add)

                if len(can_add) == 2 and len(ab[add_choice]) > len(ab[1-add_choice]):
                    add_choice = 1-add_choice

                k_choice = random.choice(list(open_sets[add_choice].dict.keys()))
                ab[add_choice][k_choice] = 1

                open_sets = [get_open(target - ab[0]*ab[1], ab[1]), get_open(target - ab[0]*ab[1], ab[0])]
                for i in range(len(open_sets)):
                    for d in ab[i].dict.keys():
                        if d in open_sets[i].dict.keys():
                            del open_sets[i].dict[d]

                # print([str(o) for o in ab])
                # print([str(o) for o in open_sets])
                # input("... ")

            if best_combo is None or sum(sum(k) for k in (ab[0]*ab[1]).dict.keys()) > best_cost:
                best_combo = ab
                best_cost = sum(sum(k) for k in (ab[0]*ab[1]).dict.keys())
            
        return best_combo, best_cost


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
    target *= t_2
    target *= target
    print("")

    ab, cost = search(target, 5)

    print("")
    print(ab[0], '\n')
    print(ab[1], '\n')
    print(ab[0]*ab[1], '\n')
    print(target, '\n')
    print(target - ab[0]*ab[1])

    print("Effective Reduction:")
    print(cost)

if __name__ == '__main__':
    main()