
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
    add_set: set
    mult_set: set
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
        poly,
        set(),
        set()
    )


class CStar:

    def __init__(self, target: SparsePoly, n_vals: int, costs={OPERATIONS.MULT: 1, OPERATIONS.ADD: 1}):
        self.target = target
        self.n = target.n
        self.target_str = str(self.target)

        self.op_costs = costs

        self.leaves = []
        for i in range(self.n):
            leaf = get_leaf(self.n, arg=i)
            self.leaves.append(leaf)
        for i in range(1, n_vals+1):
            self.leaves.append(get_leaf(self.n, val=i))
            self.leaves.append(get_leaf(self.n, val=-i))

        self.s_p = set()
        for k in self.target.dict.keys():
            elem = np.log(float(abs(self.target.dict[k]))) if self.target.dict[k] != 0 else 0
            for i in range(len(k)):
                elem += k[i]
            self.s_p.add(elem)


    def blank_circuit(self, models=[]):
        seen_ids = set()
        nodes = []

        for l in self.leaves:
            nodes.append(l)
            seen_ids.add(l.id)

        for mod in models:
            for n in mod.nodes:
                if n.id not in seen_ids and (n.id in mod.root.add_set or n.id in mod.root.mult_set):
                    seen_ids.add(n.id)
                    nodes.append(n)

        return CandidateCircuit(
            nodes[random.randrange(len(nodes))],
            0,
            nodes
        )


    def get_pred(self, circuit: CandidateCircuit, simple=False):
        r = circuit.root.poly

        q = self.target - r
        if simple:
            return sum([1 if co!=0 else 0 for co in q.dict.values()])
        d_plus = sum([abs(co) for co in q.dict.values()])

        s_a = set()
        for k in r.dict.keys():
            elem = np.log(float(abs(r.dict[k]))) if r.dict[k] != 0 else 0
            for i in range(len(k)):
                elem += k[i]
            s_a.add(elem)

        d_x = 1 - sum(self.s_p.intersection(s_a))/sum(self.s_p | s_a)

        cost = self.op_costs[OPERATIONS.ADD]*d_plus + self.op_costs[OPERATIONS.MULT]*d_x

        return cost


    def action(self, circuit: CandidateCircuit, op: OpInfo, operands: list, track_sets=True):
        if len(operands) != 2:
            raise ValueError("invalid number of operands")
        
        id = random.randint(1, 2**60)

        add_set = set()
        mult_set = set()

        if track_sets:
            add_set = operands[0].add_set | operands[1].add_set
            mult_set = operands[0].mult_set | operands[1].mult_set
            if op == OPERATIONS.ADD:
                add_set.add(id)
            else:
                mult_set.add(id)

        new_node = CandidateNode(
            op,
            False,
            -1,
            0,
            operands,
            id,
            op.func(operands[0].poly, operands[1].poly),
            add_set,
            mult_set
        )

        cost = circuit.cost + self.op_costs[op]
        if track_sets:
            cost = (
                self.op_costs[OPERATIONS.ADD]*len(new_node.add_set) +
                self.op_costs[OPERATIONS.MULT]*len(new_node.mult_set)
            )
        return CandidateCircuit(
            new_node,
            cost,
            circuit.nodes + [new_node]
        )

    
    def sample_search(self, max_iters, max_cost, alpha, gamma, verbose=True, use_pred=True, n_models=0, wrapped=False):

        best_solution = None
        solutions_found = 0

        curr = self.blank_circuit()
        prev_pred = self.get_pred(curr)

        models = []

        # verbose stuff
        last_time = time.time_ns()
        for iter in range(1, max_iters+1):

            if verbose:
                new_time = time.time_ns()
                if iter == 1 or iter == max_iters or (new_time-last_time)*1e-9 >= 1:
                    last_time = new_time
                    print("\niteration:", str(iter)+"/"+str(max_iters))
                    print("best model:", "None" if len(models) == 0 else models[0].circuit.root.poly)
                    print("solution:", None if best_solution is None else best_solution.cost)
                    print("solutions found:", solutions_found)
                    print("current cost:", curr.cost)
                    print("current prediction:", prev_pred)
                    print(curr.root.poly)
                    print("("+self.target_str+")")

            circs = []
            preds = []
            weights = []

            potential_models = []

            # just always using root for this now
            for op2 in range(len(curr.nodes)):
                for oper in [OPERATIONS.ADD, OPERATIONS.MULT]:
                    
                    new_circ = self.action(curr, oper, [curr.root, curr.nodes[op2]], track_sets=(n_models>0))

                    if max([0] + [abs(val) for val in new_circ.root.poly.dict.values()]) == 0:
                        continue

                    if new_circ.root.poly == self.target:
                        if best_solution is None or new_circ.cost < best_solution.cost:
                            best_solution = new_circ
                            if n_models > 0:
                                potential_models.append(PrioritizedCircuit(
                                    new_circ.cost, new_circ, 0
                                ))
                        solutions_found += 1

                    elif not (
                            new_circ.cost >= max_cost or
                            best_solution != None and new_circ.cost >= best_solution.cost-1
                        ):

                        pred = self.get_pred(new_circ) if use_pred else 1.5
                        circs.append(new_circ)
                        preds.append(pred)
                        weights.append(1/(new_circ.cost+alpha*pred)**gamma)

                        priority = new_circ.cost+1000*self.get_pred(new_circ, simple=True) if wrapped else new_circ.cost+alpha*pred
                        if n_models > 0 and (len(models) < n_models or priority < models[-1].priority):
                            potential_models.append(PrioritizedCircuit(
                                priority,
                                new_circ, pred
                            ))
                            
            if len(potential_models) > 0 and n_models > 0:
                new_models = models + potential_models
                new_models.sort()
                models = []
                seen_pri = set()
                for i in range(min(n_models, len(new_models))):
                    if new_models[i].priority not in seen_pri:
                        seen_pri.add(new_models[i].priority)
                        models.append(new_models[i])

            if len(circs) == 0:
                curr = self.blank_circuit([m.circuit for m in models])
            
            else:
                choice = random.choices([i for i in range(len(circs))], weights=weights, k=1)[0]
                curr = circs[choice]
                prev_pred = preds[choice]

        if wrapped:
            return solutions_found > 0, best_solution if best_solution != None else models[0].circuit
        return best_solution


    def add_circuits(self, a, b):
        id = random.randint(1, 2**60)

        add_set = set()
        mult_set = set()

        add_set = a.root.add_set | b.root.add_set
        mult_set = a.root.mult_set | b.root.mult_set
        add_set.add(id)

        new_node = CandidateNode(
            OPERATIONS.ADD,
            False,
            -1,
            0,
            [a.root, b.root],
            id,
            OPERATIONS.ADD.func(a.root.poly, b.root.poly),
            add_set,
            mult_set
        )

        cost = (
            self.op_costs[OPERATIONS.ADD]*len(new_node.add_set) +
            self.op_costs[OPERATIONS.MULT]*len(new_node.mult_set)
        )

        seen_ids = set()
        nodes = []

        for l in self.leaves:
            nodes.append(l)
            seen_ids.add(l.id)

        for mod in [a, b]:
            for n in mod.nodes:
                if n.id not in seen_ids and (n.id in mod.root.add_set or n.id in mod.root.mult_set):
                    seen_ids.add(n.id)
                    nodes.append(n)

        return CandidateCircuit(
            new_node,
            cost,
            nodes + [new_node]
        )


    def recursive_search(self, max_depth, max_iters_per, max_cost, alpha, gamma, n_models=0, verbose=True):

        old_target = self.target
        old_target_str = self.target_str
        old_s_p = self.s_p

        curr_solution = None
        found = False

        for depth in range(0, max_depth):
            if verbose:
                print("\n --- Depth", depth, " --- ")
                print(" ->", ("None" if curr_solution == None else str(curr_solution.root.poly)) + '\n')

            found, subsolution = self.sample_search(max_iters_per, max_cost, alpha, gamma, verbose=verbose, n_models=n_models+1, wrapped=True)

            if curr_solution == None:
                curr_solution = subsolution
            else:
                old_score = self.get_pred(curr_solution, simple=True)
                temp_solution = self.add_circuits(curr_solution, subsolution)
                temp_score = self.get_pred(temp_solution, simple=True)
                if  temp_score < old_score or (temp_score == old_score and self.get_pred(temp_solution) < self.get_pred(curr_solution)):
                    curr_solution = temp_solution
        
            if found:
                break

            self.target = old_target - curr_solution.root.poly
            self.target_str = str(self.target)
            self.s_p = set()
            for k in self.target.dict.keys():
                elem = np.log(float(abs(self.target.dict[k]))) if self.target.dict[k] != 0 else 0
                for i in range(len(k)):
                    elem += k[i]
                self.s_p.add(elem)

        self.target = old_target
        self.target_str = old_target_str
        self.s_p = old_s_p

        return curr_solution if found else None

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
    target *= target
    
    engine = CStar(target, 4)
    sol = engine.recursive_search(10, 8000, 20, 5, 3, n_models=1)

    print("\nTarget:", target)
    if sol == None:
        print("NO SOLUTION FOUND.\n")
        exit()
    print("Solution:", sol.root.poly)
    print("Cost:", sol.cost, "\n")

if __name__ == '__main__':
    main()