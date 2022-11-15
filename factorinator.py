
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


    common_d = tuple([min([k[i] for k in clean.dict.keys()]) if len(clean.dict) > 0 else 0 for i in range(clean.n)])

    final = SparsePoly(poly.n)

    for k in clean.dict.keys():
        adjusted = tuple([k[i] - common_d[i] for i in range(len(common_d))])
        final[adjusted] = clean[k]

    common_d = SparsePoly(poly.n, {common_d: 1})
    return final, common_d, outliers


class Factorinator:

    def __init__(self, target: SparsePoly, costs={OPERATIONS.MULT: 1, OPERATIONS.ADD: 1}):
        self.target = target
        self.n = target.n
        self.target_str = str(self.target)
        self.max_order = [max([k[i] for k in self.target.dict.keys()]) if len(self.target) > 0 else 0 for i in range(self.n)]

        self.op_costs = costs


    def cost(self, p: SparsePoly):

        cleaned = make_clean(p, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])[0]

        cost = 0

        for k in cleaned.dict.keys():
            if cleaned.dict[k] != 0:
                cost += sum(k)
                cost += np.log(float(abs(cleaned.dict[k])))

        return cost


    def total_cost(self, polys):
        cost = 0
        for poly in polys:
            cost += self.cost(poly)**2
        
        return cost


    def search(self, iters):

        ab = [None, None]
        ab[0] = SparsePoly(self.n)
        ab[0][[random.randint(0, 1+self.max_order[_]//2) for _ in range(self.n)]] = 1
        ab[1] = SparsePoly(self.n)
        ab[1][[random.randint(0, 1+self.max_order[_]//2) for _ in range(self.n)]] = 1
        c = self.target - (ab[0]*ab[1])

        best = [ab[0].copy(), ab[1].copy(), c.copy()]
        prev_cost = self.total_cost([ab[0], ab[1], c])
        best_cost = prev_cost
        k = tuple([random.randint(0, 1+self.max_order[_]//2) for _ in range(self.n)])
        momentum = 0
        rejected = False

        for iter in range(iters):
            changer = random.randrange(2)

            delta = SparsePoly(self.n)
            if rejected:
                k = tuple([random.randint(0, 1+self.max_order[_]//2) for _ in range(self.n)])

            if momentum == 0:
                if k in ab[changer].dict.keys():
                    momentum = random.choice([i for i in range(1, 4)] + [-i for i in range(1, 2)])
                else:
                    momentum = random.choice([i for i in range(1, 4)])
            delta[k] = momentum
             
            if random.random() < 0.01:
                delta -= ab[changer]
            ab[changer] += delta

            c = self.target - (ab[0]*ab[1])
            new_cost = self.total_cost([ab[0], ab[1], c])

            not_zero = sum([abs(val) for val in ab[changer].dict.values()]) > 0

            if not_zero and new_cost < best_cost:
                best = [ab[0].copy(), ab[1].copy(), c.copy()]
                best_cost = new_cost

            # print("\nd:", delta)
            # print("\na:", ab[0])
            # print("\nb:", ab[1])
            # print("\nc:", c)
            # input()
            # print(new_cost, prev_cost)
            if new_cost > prev_cost:
                rejected = True
                momentum = 0
            else:
                rejected = False

            if not_zero and (new_cost < prev_cost or random.random() < 0.01):
                prev_cost = new_cost
            else:
                ab[changer] -= delta

        # print("Target Cost:", self.cost(self.target))
        # print("Solution Cost:", np.sqrt(best_cost))
        return best


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
    
    # print("\nTarget:", target)
    # clean, common_d, outliers = make_clean(target, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    # print("\nClean:", clean)
    # print("\nCommon D:", common_d)
    # print("\nOutliers:", outliers)
    # exit()

    engine = Factorinator(make_clean(target)[0])
    sol = engine.search(10000)

    for der in sol:
        print("")
        print(der, "--->", make_clean(der)[0])
    print("-----------------------------")

    sol_cost = 0
    for s in sol:
        sengine = Factorinator(make_clean(s)[0])
        der = sengine.search(10000)
        for d in der:
            print("")
            print(d, "--->", make_clean(d)[0])
            sol_cost += sengine.cost(d)**2
    
    print("")
    print("Initial Cost:", engine.cost(target))
    print("Intermediate Cost:", sum([engine.cost(s)**2 for s in sol])**0.5)
    print("Solution Cost:", sol_cost**0.5)
    print(target, "--->", make_clean(target)[0])
    exit()

    print("\nTarget:", target)
    print("\na:", sol[0])
    print("\nb:", sol[1])
    print("\nc:", sol[2])
    print("")

if __name__ == '__main__':
    main()