
from sparse_poly import SparsePoly, LETTERS

import multivar_horner

import numpy as np
from dataclasses import dataclass, field
import heapq
import random
import time


def log_cost(k):
    # log cost heuristic of monomial
    return np.sum(np.ceil(np.log2(1+np.abs(k))))

def singlet(s: set):
    # Get the 'first' element in set
    return list(s)[0]


@dataclass(order=True)
class PrioritizedGroup:
    # wrapper for set with priority
    priority: float = field(compare=True)
    s: set = field(compare=False)

def get_prioritized_group(s: set):
    # create a set that prioritizes those with smallest exponent sum first
    return PrioritizedGroup(
        sum([np.sum(k.priv) for k in s]),
        s
    )

@dataclass(order=True)
class PrioritizedMem:
    # wrapper for array with priority
    priority: float = field(compare=True)
    s: np.ndarray = field(compare=False)

def get_prioritized_mem(k: np.ndarray):
    # get an array that prioritizes smallest sum first
    return PrioritizedMem(
        np.sum(k),
        k
    )


class Nom:
    def __init__(self, a: np.ndarray):
        """
        Hashable wrapper for array (could probably be depreciated).
        """
        self.pub = a.copy() # the difference between the true internal monomial and internal target
        self.priv = a.copy() # the true internal monomial reresented by this
        self.target = np.zeros_like(a, dtype=np.int32) # an internal target that is known
        self.target_score = 0 # cost of getting to internal target

    def check(self, pot: np.ndarray):
        # check whether potential new target is better than current, and save if true
        diff = self.priv - pot
        if np.min(diff) < 0:
            # not valid
            return
        if sum(pot) > self.target_score:
            # better, so save
            self.pub = self.priv - pot
            self.target = pot.copy()
            self.target_score = sum(pot)

    def copy(self):
        # deep copy this monomial
        n = Nom(self.priv)
        n.priv = self.priv.copy()
        n.pub = self.pub.copy()
        n.target = self.target.copy()
        n.target_score=  self.target_score

    def peek_change(self, pot: np.ndarray):
        # check the new score of the potential target without saving
        diff = self.priv - pot
        if np.min(diff) < 0:
            # not valid
            return None
        return sum(pot)

    def valid_target(self):
        # check whether the internal target is still valid after modifiying priv
        diff = self.priv - self.target
        if np.min(diff) < 0:
            # not valid
            self.target = np.zeros_like(self.target)
            self.target_score = 0
            self.pub = self.priv.copy()
            return False
        return True

    def sub(self, k: np.ndarray):
        # divide the monomial by sum divisor, updating internal and public states
        self.pub -= k
        self.priv -= k

    def __hash__(self):
        # hash using tuple
        return hash(tuple(self.priv))
    def __eq__(self, other):
        # compare using private
        return np.array_equal(self.priv, other.priv)


def caching_horners(poly: SparsePoly, verbose:bool=False, care_about_add=False, max_mem=2000, expensive_heuristic=True, div_init_mem=True, analyze_ones=False, test=False):
    """
    Create a multivariate horner scheme of the polynomial with aggresive CSE and recursive sub-polynomial optimization.
    
    Parameters:
        poly: SparsePoly polynomial describing the problem
        verbose: Whether to print verbos output
        care_about_add: Whether to factor additions into the cost function (default: False)
        max_mem: Clip the number of memoized expressions to loop through during common denominator search (default: 2000)
        expensive_heuristic: Use a slightly more complicated heuristic function to evaluate common denominators. Is slightly slower but usually gives better results (default: True)
        div_init_mem: Whether to initialze all mononomial expressions up to max_order/2 rather than max_order. True usually works better. better. (default: True)
        analyze_ones: Search over all memoized exxpressions to reduced monomials to nearest memoized neighbor rather than 0. Slightly better results but takes much longer to run (default: False)
    
    Returns:
        cost: The number of multiplications (and additions if specified) of the solution.
    """

    # number of variables
    n = poly.n

    # set of Noms that represent the problem (Nom reperesents array with a_j[i] = exponent of x_i in monomial_j)
    problem = set()
    for k in poly.dict.keys():
        a = np.array(k, dtype=np.int32)
        problem.add(Nom(a))

    # keep track of cost
    cost = 0

    # ordered queue of polynomials to be decomposed, with smallest first
    groups = []
    heapq.heapify(groups)
    heapq.heappush(groups, get_prioritized_group(problem))

    # list of all found and chached/memoized monomials, sorted smallest to largest
    mem_base = [get_prioritized_mem(np.zeros(n, dtype=np.int32))]
    mem = []
    heapq.heapify(mem)
    # set of all found monomials, represented as tuples
    mem_str = set([tuple(np.zeros(n, dtype=np.int32))])

    def mem_update(nommer: Nom, base=False):
        # Add a monomial to mem.
        nonlocal groups
        nonlocal mem
        nonlocal test

        k = nommer.priv.copy()
        if tuple(k) not in mem_str and (base or not test):
            # for group in groups:
            #     for p in group.s:
            #         p.check(k)
            if not base:
                heapq.heappush(mem, get_prioritized_mem(k))
            else:
                mem_base.append(get_prioritized_mem(k))
            mem_str.add(tuple(k))

    # initialize mem with all the singe-variable monomials up to some order
    for i in range(n):
        for m in range(1, 2 if test else 1+round(np.ceil(poly.max_order()/(2 if div_init_mem else 1)))):
            mon = np.zeros(n, dtype=np.int32)
            mon[i] = m
            mem_update(Nom(mon), base=True)
            cost += 1 if m > 1 else 0

    def clean(s: set, d: np.ndarray=np.zeros(n, dtype=np.int32)):
        """
        Remove known monomials and divide by common denominator.
        s: grouup set to reduce
        d: array to use as common denominator (default zeros just removes known components without dividing)
        """
        nonlocal cost
        nonlocal care_about_add
        if len(s) == 0:
            return set()

        # if this is monomial and we are reducing it, then cache
        if len(s) == 1 and np.sum(d) > 0:
            mem_update(singlet(s))

        # reduce with common denominator
        if np.sum(d) > 0:
            cost += 1 # this represents a multiplication
            for k in s:
                k.sub(d)
        # for k in s:
        #     if not k.valid_target():
        #         for m in mem:
        #             k.check(m)

        # cache = None
        # if len(s) == 1:
        #     cache = singlet(s)

        # check all monomials to see if they are known
        kept = set()
        fence = False
        for k in s.copy():
            if tuple(k.priv) in mem_str or sum(k.priv) <= 1:
                # is known
                if np.sum(k.priv) >= 1: # and not test:
                    # this is not a scalar, so take into account coefficient
                    cost += 1
                if care_about_add and fence:
                    # this gets added
                    fence = True # fencepost bug
                    cost += 1
            else:
                # not known, continue to reduce it
                kept.add(k)
    
        # if len(kept) == 0 and cache is not None:
        #     mem_update(cache)

        return kept

    # stuff for printing
    init_size = len(groups[0].s)
    init_time = time.time()

    # go until we have solved every sub-polynomial
    while len(groups) > 0:

        # pop smallest remaining group, and clean without d to check if any monomials have been found since pushing
        before = heapq.heappop(groups).s
        curr_group = clean(before)
        if len(curr_group) == 0:
            continue
        
        # print stuff
        if verbose:
            if len(curr_group) > 1 or test:
                num_left = sum([len(g.s) for g in groups]) + len(curr_group)
                p_left = sum([sum([sum(elem.priv) for elem in g.s]) for g in groups]) + sum([sum(elem.priv) for elem in curr_group])
                print("Remaining:", num_left, '&', p_left, "("+str(len(curr_group))+' & '+str(sum([sum(elem.priv) for elem in curr_group]))+")", " --  Cost:", cost, " --  Memory Size:", len(mem), " --  Est. Time Left:", round(num_left * (time.time()-init_time)/(1+init_size-num_left)), "s")

        # list of common denominators that we will check
        common_list = [mem_base[m].s for m in range(min(len(mem_base), max_mem))]
        if len(common_list) < max_mem:
            common_list += [mem[m].s for m in range(min(len(mem), max_mem-len(common_list)))]

        # weights of how good each common denom is
        weights = [0 for _ in range(len(common_list))]

        # iterate through all commmon denoms
        for g in range(len(common_list)):
            if np.sum(common_list[g]) == 0:
                # this guy sneaks in but can't be used
                continue

            found_sols = 0 # number of monomials that are solved completely by this denom
            check_sol = len(mem) >= len(curr_group) # whether to check for found_sols

            if len(curr_group) > 1 or not analyze_ones:
                # usual loop
                score = np.sum(common_list[g]) # cache this for speed

                for k in curr_group:
                    # loop though all monomials in group
                    diff = k.priv - common_list[g] # this is what is left after reduction by commons[g]
                    if np.min(diff) >= 0:
                        # if k is divisable by the commons[g] 

                        # add to weight according to heuristic
                        weights[g] +=  score if not expensive_heuristic else np.sum(common_list[g] / np.maximum(1, k.priv)**(0.5))

                        # if reduced to solution, keep track
                        if check_sol and tuple(diff) in mem_str:
                            found_sols += 1

            else:
                # special loop for aggresive optimization of single monomials
                for k in curr_group:
                    diff = k.priv - common_list[g] # this is what is left after reduction by commons[g]
                    if np.min(diff) >= 0:
                        # if k is divisable by the commons[g] 

                        if check_sol and tuple(diff) in mem_str:
                            # if this is a sol, we don't need to search further
                            found_sols += 1
                            weights[g] = 100
                            break

                        # loop through every monomial in mem to find closest neighor to diff
                        cop = Nom(diff)
                        for m in mem:
                            cop.check(m.s)

                        # score based on how close to closest neighbor diff gets.
                        # this causes descent towards that neighbor instead of zero
                        weights[g] += 1/(1+np.sum(cop.pub))

            # if this denom gives solution for every monomial in group, then can't do better and can stop searching
            if found_sols == len(curr_group):
                break

        # get the best common denominator
        common = common_list[weights.index(max(weights))]

        # set that can and will be reduced
        reduce = set()
        # set that cannot and will not be reduced
        keep = set()
        for k in curr_group:
            diff = k.priv - common
            if min(diff) >= 0:
                # is divisable by common
                reduce.add(k)
            else:
                # is not divisible by common
                keep.add(k)

        if len(keep) > 0:
            # if there are monomials that cannot be reduced here, they split off into their own group
            if care_about_add:
                # reduce and keep are linked by an addition
                cost += 1
            heapq.heappush(groups, get_prioritized_group(keep))
        
        # verbose stuff
        bef = None
        if verbose:
            bef = (sum([sum(elem.priv) for elem in reduce]), len(reduce))

        # get the resulting group from reducing reduce with common
        after_reduce = clean(reduce, common)

        # verbose stuff
        if verbose:
            if len(curr_group) > 1 or test:
                aft = (sum([sum(elem.priv) for elem in after_reduce]), len(after_reduce))
                print(" -> Common:", tuple(common), " -- Reduction:", bef[1], '&', bef[0], " -- After:", aft[1], '&', aft[0])

        # if there is still more to reduce, push the remaining group to stack
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

    # generate some big random polynomial
    N = 5
    scale = 3
    target = SparsePoly(N)
    more = 1000
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
    print("created benchmark...\n")

    # get solution using our method
    cost = caching_horners(target, verbose=False)
    test_cost = caching_horners(target, verbose=False, test=True)

    print("\n --> Cost:", cost)
    print(" --> Test Cost:", test_cost)
    print("")

    # the most basic representation just multiplies and adds every monomial one by one
    print("Naive Estimate:", sum([1+np.sum(np.maximum(0, np.array(k)-1)) for k in target.dict.keys()])-1)

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

    print()

    # show the polynomial we computed
    # print(target)
    print("")


if __name__ == '__main__':
    main()