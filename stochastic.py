
import numpy as np
import numpy.polynomial as poly
from sklearn.metrics import mean_squared_error

from super_circuit import SuperCircuit

"""
Framework for potential stochastic search algorithm.
"""

def poly_MSE(a_poly, b_poly):
    """
    Mean squared error between the coefficients of poly.Polynomials.
    If one polynomial is lower degree, its coefficients are zero-padded.
    """

    a_size = a_poly.degree() + 1
    b_size = b_poly.degree() + 1
    max_n = max(a_size, b_size)

    a, b = np.zeros(max_n), np.zeros(max_n)
    a[:a_size] = a_poly.coef
    b[:b_size] = b_poly.coef

    return mean_squared_error(a, b)

class SearchEngine:

    def __init__(self, circuit: SuperCircuit):
        self.circuit = circuit.copy()

    def search(self, max_iters):

        # keep track of the best solution that we have found
        best_solution = None

        # do iterations
        for iter in range(max_iters):

            # try making a change to the node
            change = create_some_change
            valid = circuit.do_change(change) # if False change was invalid and not done

            # make sure that this change is valid
            if valid:

                # check whether we want to keep this change, according to the metaheuristic
                accept = flip_accept_coin(self.circuit.cost)
                if accept:

                    # check is we have found a new best solution
                    if best_solution == None or (self.circuit.is_solution and self.circuit.cost < best_solution.cost):
                        best_solution = self.circuit.copy()
                
                else:

                    # undo the last change because it was bad
                    self.circuit.undo_change(change)

        return best_solution

def main():
    a = poly.Polynomial([1, 2])
    b = poly.Polynomial([1, 3])
    print(poly_MSE(a, b))

if __name__ == '__main__':
    main()