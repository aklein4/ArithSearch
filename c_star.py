
from sparse_poly import SparsePoly, LETTERS, poly_MSE
from circuit import OpInfo, OPERATIONS

import numpy as np
from dataclasses import dataclass, field
import heapq
import random
import time


@dataclass
class CandidateNode:
    op: OpInfo
    is_leaf: bool
    arg: int
    val: int
    operands: list
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


def get_leaf(n_args, arg=-1, val=0):
    if (arg == -1) == (val == 0):
        raise ValueError("exactly one of arg or val must be set in get_leaf")
    
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
        [],
        random.randint(1, 2**60),
        poly
    )


class CStar:

    def __init__(self, target: SparsePoly, n_vals: int, costs={OPERATIONS.MULT: 1, OPERATIONS.ADD: 1}):
        self.target = target
        self.n = target.n
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

        self.s_p = set()
        for k in self.target.dict.keys():
            elem = np.log(self.target.dict[k]) if self.target.dict[k] != 0 else 0
            for i in range(len(k)):
                elem += k[i]
            self.s_p.add(elem)


    def blank_circuit(self):
        return CandidateCircuit(
            self.basic_leaf,
            0,
            self.leaves.copy()
        )


    def get_pred(self, circuit: CandidateCircuit):
        r = circuit.root.poly

        q = self.target - r
        d_plus = sum([abs(co) for co in q.dict.values()])

        s_a = set()
        for k in r.dict.keys():
            elem = np.log(r.dict[k]) if r.dict[k] != 0 else 0
            for i in range(len(k)):
                elem += k[i]
            s_a.add(elem)

        d_x = 1 - sum(self.s_p.intersection(s_a))/sum(self.s_p | s_a)

        # # matrix of target log chunks
        # K_p = np.zeros((len(self.target), 1+self.n))
        # k_ind = 0
        # for k in self.target.dict.keys():
        #     K_p[k_ind] = np.log(self.target.dict[k] if self.target.dict[k] != 0 else 0.01)
        #     for i in range(len(k)):
        #         K_p[k_ind][i+1] = k[i]
        #     k_ind += 1
        
        # # matrix of a log chunks
        # K_a = np.zeros((len(r), 1+self.n))
        # k_ind = 0
        # for k in r.dict.keys():
        #     K_a[k_ind] = np.log(r.dict[k] if r.dict[k] != 0 else 0.01)
        #     for i in range(len(k)):
        #         K_a[k_ind][i+1] = k[i]
        #     k_ind += 1
        
        # # get closest candidate for each k in target
        # d_x_list = np.zeros(K_p.shape[0])
        # for p in range(K_p.shape[0]):
        #     min_d = 0

        #     # loop through each k in a to find min
        #     for a in range(K_a.shape[0]):
        #         this_d = (self.n+1) * mean_squared_error(K_p[p], K_a[a])
        #         min_d += this_d

        #     d_x_list[p] = min_d/K_a.shape[0]

        # d_x = np.average(d_x_list)

        cost = self.op_costs[OPERATIONS.ADD]*d_plus + self.op_costs[OPERATIONS.MULT]*d_x

        return cost


    def action(self, circuit: CandidateCircuit, op: OpInfo, operands: list):
        if len(operands) != 2:
            raise ValueError("invalid number of operands")
        
        op_list = operands

        new_node = CandidateNode(
            op,
            False,
            -1,
            0,
            operands,
            random.randint(1, 2**60),
            op.func(op_list[0].poly, op_list[1].poly)
        )

        return CandidateCircuit(
            new_node if circuit.root.id in [oper.id for oper in operands] or circuit.root.is_leaf else circuit.root,
            circuit.cost + self.op_costs[op],
            circuit.nodes + [new_node]
        )

    
    def sample_search(self, max_iters, gamma, verbose=True, max_cost=10, use_pred=True):

        best_solution = None

        blank = self.blank_circuit()
        curr = blank
        blank_pred = self.get_pred(curr)
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
                    print("current prediction:", prev_pred)
                    print(curr.root.poly)
                    print("("+self.target_str+")")

            circs = []
            preds = []
            weights = []

            seen = set()
            for op1 in range(len(curr.nodes)):
                # just always using root for this now
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
                                pred = self.get_pred(new_circ) if use_pred else 1.5
                                circs.append(new_circ)
                                preds.append(pred)
                                weights.append(1/(1+pred)**gamma)
            
            choice = random.choices([i for i in range(len(circs))], weights=weights, k=1)[0]
            curr = circs[choice]
            prev_pred = preds[choice]

            if curr.cost > max_cost or best_solution != None and curr.cost >= best_solution.cost-1:
                curr = blank
                blank_pred = blank_pred

        return best_solution


    def queue_search(self, max_iters, alpha, verbose=True, max_cost=10, use_pred=True):
        q = []
        base_circuit = self.blank_circuit()
        heapq.heappush(q, PrioritizedCircuit(
            self.get_pred(base_circuit),
            base_circuit,
            self.get_pred(base_circuit)
        ))

        best_solution = None
        solutions_found = 0

        last_time = time.time_ns()
        for iter in range(1, max_iters+1):

            # unpack from queue
            curr_item = heapq.heappop(q)
            curr_pred = curr_item.priority
            curr = curr_item.circuit
            curr_dist = curr_item.dist

            # verbose stuff
            if verbose:
                new_time = time.time_ns()
                if iter == 1 or iter == max_iters or (new_time-last_time)*1e-9 >= 0.25:
                    last_time = new_time
                    print("\niteration:", str(iter)+"/"+str(max_iters))
                    print("solution:", None if best_solution is None else best_solution.cost)
                    print("solutions found:", solutions_found)
                    print("current cost:", curr.cost)
                    print("current dist:", curr_dist)
                    print("current prediction:", curr_pred)
                    print(curr.root.poly)
                    print("("+self.target_str+")")

            seen = set()
            for op1 in range(len(curr.nodes)):
                # just using root for this now
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
                            elif not (
                                    new_circ.cost >= max_cost or
                                    best_solution != None and new_circ.cost >= best_solution.cost-1
                                ):
                                pred = self.get_pred(new_circ)
                                heapq.heappush(q, PrioritizedCircuit(
                                    new_circ.cost + alpha*pred,
                                    new_circ,
                                    pred
                                ))
            
        
            if len(q) > 5*max_iters:
                clean_q = []
                done = True
                for i in range(max_iters):
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
    target += t_2*2 + 2
    
    engine = CStar(target, 4)
    sol = engine.queue_search(100000, 10, max_cost=15, use_pred=True)

    print("\nTarget:", target)
    if sol == None:
        print("NO SOLUTION FOUND.\n")
        exit()
    print("Solution:", sol.root.poly)
    print("Cost:", sol.cost, "\n")

if __name__ == '__main__':
    main()