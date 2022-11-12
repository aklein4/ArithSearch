
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
    
    return CandidateNode(
        OPERATIONS.LEAF,
        True,
        arg,
        val,
        set(),
        scaler,
        grad,
        H,
        random.randint(1, 2**60)
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
    return abs(a[0] - b[0]) + np.linalg.norm(a[1] - b[1]) + np.linalg.norm(a[2] - b[2])

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
        r = circuit.root

        root_dist = compare_tensors(self.target_tensors, (r.scalar, r.grad, r.H))
        avg_size = sum([
        #     tensor_size((n.scalar, n.grad, n.H)) for n in circuit.nodes
        ]) / max(1, len(circuit.nodes))
        not_leaves = max(1, sum([1 if not n.is_leaf else 0 for n in circuit.nodes]))

        return circuit.cost + gamma*abs(root_dist)


    def action(self, circuit: CandidateCircuit, op: OpInfo, operands: list):
        if len(operands) != 2:
            raise ValueError("invalid number of operands")
        
        op_list = operands
        scalar, grad, H = None, None, None
        if op is OPERATIONS.MULT:
            scalar, grad, H = mult_nodes(op_list[0], op_list[1])
        else:
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
            random.randint(1, 2**60)
        )

        return CandidateCircuit(
            new_node if circuit.root.id in [oper.id for oper in operands] or circuit.root.is_leaf else circuit.root,
            circuit.cost + self.op_costs[op],
            circuit.nodes + [new_node]
        )

    
    def search(self, max_iters, gamma, verbose=True):
        q = []
        base_circuit = self.blank_circuit()
        heapq.heappush(q, PrioritizedCircuit(
            self.get_pred(base_circuit, gamma),
            base_circuit,
            -1
        ))

        best_solution = None

        # verbose stuff
        last_time = time.time_ns()
        for iter in range(1, max_iters+1):
            curr_item = heapq.heappop(q)
            if len(q) > 1 and random.random() < 0.5:
                curr_item = q.pop(random.randrange(0, len(q)))
            curr_pred = curr_item.priority
            curr = curr_item.circuit
            curr_dist = curr_item.dist

            if verbose:
                new_time = time.time_ns()
                if iter == 1 or iter == max_iters or (new_time-last_time)*1e-9 >= 0.25:
                    last_time = new_time
                    print("\niteration:", str(iter)+"/"+str(max_iters))
                    print("solution:", None if best_solution is None else best_solution.cost)
                    print("current cost:", curr.cost)
                    print("current dist:", curr_dist)
                    print("current prediction:", curr_pred)
                    print(get_poly(curr.root))
                    print("("+self.target_str+")")

            seen = set()
            for op1 in range(len(curr.nodes)):
                for op2 in range(len(curr.nodes)):
                    for oper in [OPERATIONS.ADD, OPERATIONS.MULT]:
                        fset = frozenset([op1, op2, oper])
                        if fset not in seen:

                            seen.add(fset)
                            new_circ = self.action(curr, oper, [curr.nodes[op1], curr.nodes[op2]])
                            d = compare_tensors(
                                (new_circ.root.scalar, new_circ.root.grad, new_circ.root.H),
                                self.target_tensors
                            )

                            if d == 0:
                                if best_solution is None or new_circ.cost < best_solution.cost:
                                    best_solution = new_circ
                            else:
                                heapq.heappush(q, PrioritizedCircuit(
                                    self.get_pred(new_circ, gamma),
                                    new_circ,
                                    d
                                ))
            
            # fake = self.blank_circuit()
            # heapq.heappush(q, PrioritizedCircuit(
            #                         self.get_pred(fake, gamma),
            #                         fake,
            #                         d
            #                     ))
        
            if len(q) > 100000:
                clean_q = []
                done = True
                for i in range(10000):
                    it = q[i]
                    if best_solution == None or it.circuit.cost < best_solution.cost:
                        done = False
                    heapq.heappush(clean_q, it)
                if done:
                    return best_solution
                q = clean_q

        return best_solution

def main():
    target = SparsePoly(3)
    target[1, 0, 0] = 1
    target[0, 1, 0] = 1
    target *= target
    t_2 = SparsePoly(3)
    t_2[0, 0, 1] = 2
    target *= t_2
    target += t_2 + 1

    engine = CStar(target, 4)
    sol = engine.search(10000, 1)

    print("\nTarget:", target)
    print("Solution:", get_poly(sol.root))
    print("Cost:", sol.cost, "\n")

if __name__ == '__main__':
    main()