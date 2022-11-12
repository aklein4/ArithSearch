
from sparse_poly import SparsePoly, LETTERS
from circuit import OpInfo, OPERATIONS

import numpy as np
from dataclasses import dataclass, field
import heapq
import random
import time

DEFAULT_EVAL = 1

@dataclass
class CandidateNode:
    op: OpInfo
    is_leaf: bool
    arg: int
    val: int
    operands: list
    scalar: int
    grad: np.ndarray
    H: np.ndarray
    id: int
    poly: SparsePoly
@dataclass
class CandidateCircuit:
    root: CandidateNode
    cost: float
    nodes: list

@dataclass(order=True)
class PrioritizedCircuit:
    priority: float = field(compare=True)
    circuit: CandidateCircuit = field(compare=False)
    dist: float = field(compare=False)


def get_leaf(n_args, eval_point=DEFAULT_EVAL, arg=-1, val=0):
    if (arg == -1) == (val == 0):
        raise ValueError("exactly one of arg or val must be set in get_leaf")
    if not isinstance(eval_point, int):
        raise ValueError("eval point must be an int")

    scaler, grad, H = (eval_point if arg!=-1 else val), np.zeros(n_args, dtype=np.int64), np.zeros((n_args, n_args), dtype=np.int64)
    if arg != -1:
        grad[arg] = 1
    
    poly = SparsePoly(n_args)
    k = [0 for _ in range(n_args)]
    if arg != -1:
        k[arg] = 1
        poly[k] = 1
    else:
        poly[k] = val

    return CandidateNode(
        OPERATIONS.LEAF,
        True,
        arg,
        val,
        set(),
        scaler,
        grad,
        H,
        random.randint(1, 2**60),
        poly
    )


def get_poly(root: CandidateNode):
    n_args = root.grad.size

    outs = {}
    track_stack = []
    curr = root

    while True:

        if curr.is_leaf:
            if curr.arg != -1:
                s = SparsePoly(n_args)
                l = [0 for i in range(n_args)]
                l[curr.arg] = 1
                s[l] = 1
                if curr is root:
                    return s
                outs[curr.id] = s
                curr = track_stack.pop()
            else:
                s = SparsePoly(n_args)
                s += curr.val
                if curr is root:
                    return s
                outs[curr.id] = s
                curr = track_stack.pop()

        else:
            for op in curr.operands:
                if op.id not in outs.keys():
                    track_stack.append(curr)
                    curr = op
                    break
            else:
                if curr.id not in outs.keys():

                    outs[curr.id] = curr.op.func(
                        outs[curr.operands[0].id],
                        outs[curr.operands[1].id]
                    )
                    if curr is root:
                        break

                curr = track_stack.pop()

    return outs[root.id]


def mult_nodes(a: CandidateNode, b: CandidateNode):
    outer_p = np.tensordot(a.grad, b.grad, axes=0)
    return (
        a.scalar * b.scalar,
        a.scalar * b.grad + a.grad * b.scalar,
        a.H * b.scalar + outer_p + outer_p.T + a.scalar * b.H
    )

def add_nodes(a: CandidateNode, b: CandidateNode):
    return a.scalar + b.scalar, a.grad + b.grad, a.H + b.H


def analyse_poly(poly: SparsePoly, eval_point=DEFAULT_EVAL):
    if not isinstance(eval_point, int):
        raise ValueError("eval point must be an int")
        
    n = poly.n
    scalar, grad, H = 0, np.zeros(n, dtype=np.int64), np.zeros((n, n), dtype=np.int64)

    def eval_tuple(tup):
        ev = 1
        for i in range(len(tup)):
            ev *= eval_point**tup[i]
        return ev

    for k in poly.dict.keys():
        a = poly.dict[k]
        scalar += a*eval_tuple(k)

        for i in range(len(k)):
            if k[i] > 0:
                k_g = list(k)
                k_g[i] = k_g[i] - 1
                a_g = a * k[i]
                grad[i] += a_g * eval_tuple(k_g)

                for j in range(len(k)):
                    if k_g[j] > 0:
                        k_gg = k_g.copy()
                        k_gg[j] = k_gg[j] - 1
                        a_gg = a_g * k_g[j]
                        H[i][j] += a_gg * eval_tuple(k_gg)
    
    return scalar, grad, H


def compare_tensors(a, b):
    return abs(a[0] - b[0]) + np.linalg.norm(a[1] - b[1])/a[1].size + np.linalg.norm(a[2] - b[2])/a[2].size

def tensor_size(a):
    return abs(a[0]) + np.linalg.norm(a[1]) + np.linalg.norm(a[2])

class CStar:

    def __init__(self, target: SparsePoly, n_vals: int, costs={OPERATIONS.MULT: 1, OPERATIONS.ADD: 1}):
        self.target = target
        self.n = target.n
        self.target_tensors = analyse_poly(target)
        self.target_str = str(self.target)

        self.op_costs = costs

        self.leaves = []
        self.basic_leaf = None
        for i in range(self.n):
            leaf = get_leaf(self.n, arg=i)
            self.leaves.append(leaf)
            if i == 0:
                self.basic_leaf = leaf
        for i in range(1, n_vals+1):
            self.leaves.append(get_leaf(self.n, val=i))


    def blank_circuit(self):
        return CandidateCircuit(
            self.basic_leaf,
            0,
            self.leaves.copy()
        )


    def get_pred(self, circuit: CandidateCircuit, gamma: float):
        r = circuit.root.poly

        return 1/(1+sum([abs(val) for val in (self.target - r).dict.values()]))**gamma


    def action(self, circuit: CandidateCircuit, op: OpInfo, operands: list):
        if len(operands) != 2:
            raise ValueError("invalid number of operands")
        
        op_list = operands
        scalar, grad, H = None, None, None
        # if op is OPERATIONS.MULT:
        #     scalar, grad, H = mult_nodes(op_list[0], op_list[1])
        scalar, grad, H = add_nodes(op_list[0], op_list[1])

        new_node = CandidateNode(
            op,
            False,
            -1,
            0,
            operands,
            scalar,
            grad,
            H,
            random.randint(1, 2**60),
            op.func(op_list[0].poly, op_list[1].poly)
        )

        return CandidateCircuit(
            new_node if circuit.root.id in [oper.id for oper in operands] or circuit.root.is_leaf else circuit.root,
            circuit.cost + self.op_costs[op],
            circuit.nodes + [new_node]
        )

    
    def search(self, max_iters, gamma, verbose=True, max_cost=10, use_pred=True):

        best_solution = None

        blank = self.blank_circuit()
        curr = blank
        blank_pred = self.get_pred(curr, gamma)
        blank_dist = 0
        prev_dist = blank_dist
        prev_pred = blank_pred
        solutions_found = 0

        # verbose stuff
        last_time = time.time_ns()
        for iter in range(1, max_iters+1):

            if verbose:
                new_time = time.time_ns()
                if iter == 1 or iter == max_iters or (new_time-last_time)*1e-9 >= 0.25:
                    last_time = new_time
                    print("\niteration:", str(iter)+"/"+str(max_iters))
                    print("solution:", None if best_solution is None else best_solution.cost)
                    print("solutions found:", solutions_found)
                    print("current cost:", curr.cost)
                    print("current dist:", prev_dist)
                    print("current prediction:", prev_pred)
                    print(get_poly(curr.root))
                    print("("+self.target_str+")")

            circs = []
            dists = []
            preds = []
            weights = []

            seen = set()
            for op1 in range(len(curr.nodes)):
                for op2 in range(len(curr.nodes)):
                    for oper in [OPERATIONS.ADD, OPERATIONS.MULT]:
                        fset = frozenset([0, op2, oper])
                        if fset not in seen:

                            seen.add(fset)
                            new_circ = self.action(curr, oper, [curr.root, curr.nodes[op2]])

                            if new_circ.root.poly == self.target:
                                if best_solution is None or new_circ.cost < best_solution.cost:
                                    best_solution = new_circ
                                solutions_found += 1
                            else:
                                pred = self.get_pred(new_circ, gamma)
                                circs.append(new_circ)
                                dists.append(0)
                                preds.append(pred)
                                weights.append(pred if use_pred else 1.5)
            
            choice = random.choices([i for i in range(len(circs))], weights=weights, k=1)[0]
            curr = circs[choice]
            prev_dist = dists[choice]
            prev_pred = preds[choice]

            if curr.cost > max_cost:
                curr = blank
                prev_dist = blank_dist
                blank_pred = blank_pred

            # fake = self.blank_circuit()
            # heapq.heappush(q, PrioritizedCircuit(
            #                         self.get_pred(fake, gamma),
            #                         fake,
            #                         d
            #                     ))
        
            # if len(q) > 100000:
            #     clean_q = []
            #     done = True
            #     for i in range(10000):
            #         it = q[i]
            #         if best_solution == None or it.circuit.cost < best_solution.cost:
            #             done = False
            #         heapq.heappush(clean_q, it)
            #     if done:
            #         return best_solution
            #     q = clean_q

        return best_solution

def main():
    target = SparsePoly(3)
    target[1, 0, 0] = 1
    target[0, 1, 0] = 1
    target *= target
    t_2 = SparsePoly(3)
    t_2[0, 0, 1] = 2
    target *= t_2
    target += t_2
    
    engine = CStar(target, 4)
    sol = engine.search(10000, 5, max_cost=10, use_pred=True)

    print("\nTarget:", target)
    if sol == None:
        print("NO SOLUTION FOUND.\n")
        exit()
    print("Solution:", get_poly(sol.root))
    print("Cost:", sol.cost, "\n")

if __name__ == '__main__':
    main()