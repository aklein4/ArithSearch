
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import dual_annealing, differential_evolution
import itertools

LETTERS = "abcdefghijklmnopqrstuvwxyz"

class SparsePoly:

    def __init__(self, n_var, dict=None):
        self.n = n_var
        self.dict = {}
        if dict != None:
            self.dict = dict.copy()


    def copy(self):
        return SparsePoly(self.n, self.dict)

    def _zero_tup(self):
        return tuple([0 for i in range(self.n)])

    def __eq__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self.dict == {self._zero_tup(): other}
        elif isinstance(other, SparsePoly):
            return self.dict == other.dict
        else:
            raise ValueError("invalid type in __eq__")


    def __add__(self, other):
        if isinstance(other, (int, float)):
            new_p = self.copy()
            z = self._zero_tup()
            if z not in new_p.dict.keys():
                new_p.dict[z] = 0
            new_p.dict[z] += other
            if new_p.dict[z] == 0:
                new_p.dict.pop(z)
            return new_p

        elif isinstance(other, SparsePoly):
            if self.n != other.n:
                raise ValueError("cannot multiply polynomials with different numbers of variables")
            new_p = SparsePoly(self.n)
            for them in other.dict.keys():
                new_p.dict[them] = other.dict[them]
            for me in self.dict.keys():
                if me not in new_p.dict.keys():
                    new_p.dict[me] = 0
                new_p.dict[me] += self.dict[me]

            to_remove = set()
            for k in new_p.dict.keys():
                if new_p.dict[k] == 0:
                    to_remove.add(k)
            for k in to_remove:
                new_p.dict.pop(k)
            return new_p

        else:
            raise ValueError("invalid type in __add__")

    def __iadd__(self, other):
        return self + other


    def __sub__(self, other):
        if isinstance(other, (int, float)):
            new_p = self.copy()
            z = self._zero_tup()
            if z not in new_p.dict.keys():
                new_p.dict[z] = 0
            new_p.dict[z] -= other
            if new_p.dict[z] == 0:
                new_p.dict.pop(z)
            return new_p

        elif isinstance(other, SparsePoly):
            if self.n != other.n:
                raise ValueError("cannot multiply polynomials with different numbers of variables")
            new_p = SparsePoly(self.n)
            for them in other.dict.keys():
                new_p.dict[them] = -other.dict[them]
            for me in self.dict.keys():
                if me not in new_p.dict.keys():
                    new_p.dict[me] = 0
                new_p.dict[me] += self.dict[me]

            to_remove = set()
            for k in new_p.dict.keys():
                if new_p.dict[k] == 0:
                    to_remove.add(k)
            for k in to_remove:
                new_p.dict.pop(k)
            return new_p

        else:
            raise ValueError("invalid type in __add__")

    def __isub_(self, other):
        return self - other


    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_p = self.copy()
            for elem in new_p.dict.keys():
                new_p.dict[elem] *= other
            return new_p

        elif isinstance(other, SparsePoly):
            if self.n != other.n:
                raise ValueError("cannot multiply polynomials with different numbers of variables")
            new_p = SparsePoly(self.n)
            for them in other.dict.keys():
                for me in self.dict.keys():
                    tup = tuple([them[i]+ me[i] for i in range(self.n)])
                    if tup not in new_p.dict.keys():
                        new_p.dict[tup] = 0
                    new_p.dict[tup] += self.dict[me] * other.dict[them]
            return new_p

        else:
            raise ValueError("invalid type in __mul__")
    

    def __imul__(self, other):
        return self * other


    def __getitem__(self, *args):
        arg = None

        if len(args) == 1:
            failed = False
            try:
                arg = tuple(args[0])
            except:
                if self.n == 1:
                    arg = (args[0],)
                else:
                    failed = True
            if failed:
                raise ValueError("invalid argument in __getitem___")

        elif len(args) == self.n:
            arg = tuple(args)
        else:
            raise ValueError("invalid number of arguments in SparsePolynomial __getitem__")

        return self.dict[arg]


    def __setitem__(self, *args):
        arg = None

        if len(args) == 2:
            failed = False
            try:
                arg = tuple(args[0])
            except:
                if self.n == 1:
                    arg = (args[0],)
                else:
                    failed = True
            if failed:
                raise ValueError("invalid argument in __setitem___")

        elif len(args) == self.n+1:
            arg = tuple([args[i] for i in range(self.n)])
        else:
            raise ValueError("invalid number of arguments in SparsePolynomial __setitem__")

        val = args[len(args)-1]
        self.dict[arg] = val


    def __len__(self):
        return len(self.dict)


    def __str__(self) -> str:
        s = ""
        for k in self.dict.keys():
            seg = ""
            if s != "":
                seg += " + "
            seg += str(self.dict[k])
            for i in range(len(k)):
                if k[i] == 1:
                    seg += LETTERS[i]
                elif k[i] > 1:
                    seg += LETTERS[i] + "^" + str(k[i])
            s += seg
        if s == "":
            return "0"
        return s

    def max_order(self):
        max_o = 0
        for k in self.dict.keys():
            max_o = max(max_o, max(k))
        return max_o

def encode(p: SparsePoly, max_order):
    a = np.zeros(2 + (1+max_order) ** p.n, dtype=np.int64)
    a[0] = p.n
    a[1] = max_order

    for k in p.dict.keys():
        ind = 2
        for i in range(len(k)):
            ind += k[i] * max_order**i
        a[ind] = p.dict[k]
    
    return a

def decode(a: np.ndarray):
    n = a[0]
    max_order = a[1]

    p = SparsePoly(n)

    perms = itertools.product([i for i in range(max_order)], repeat=n)

    for perm in perms:
        tup = tuple(perm)
        ind = 2
        for i in range(len(perm)):
            ind += tup[i] * max_order**i
        
        if a[ind] != 0:
            p.dict[tup] = a[ind]

    return p

def poly_MSE(p1: SparsePoly, p2: SparsePoly):
    all_keys = set(p1.dict.keys()) | set(p2.dict.keys())

    l1 = []
    l2 = []
    for k in all_keys:
        l1.append(p1.dict.get(k, 0))
        l2.append(p2.dict.get(k, 0))

    return mean_squared_error(np.array(l1), np.array(l2))

def _check_div_answer(vec, *args):
    p1 = args[0]
    p2 = args[1]
    p_delt = decode(np.round_(vec).astype(np.int64))

    p_check = p2 * p_delt

    for k in p_check.dict.keys():
        for k_2 in p1.dict.keys():
            good = False
            for i in range(len(k)):
                if k[i] <= k_2[i]:
                    good = True
                    break
            if not good:
                return 1000

    return len((p1-p_check).dict)

def poly_div(p1: SparsePoly, p2: SparsePoly):
    if p1.n != p2.n:
        raise ValueError("Cannot divide polynomials of different n")

    max_order = max(p1.max_order(), p2.max_order())
    max_val = max(max(p1.dict.values()), max(p2.dict.values()))

    output = dual_annealing(
            _check_div_answer,
            [(p1.n, p1.n+.001), (max_order, max_order+.001)] + [(0, max_val) for _ in range((1+max_order) ** p1.n)],
            args = (p1, p2),
            no_local_search=True
            #x0=encode(p2, max_order)
        )
    
    return decode(np.round_(output.x).astype(np.int64))

def main():
    target = SparsePoly(3)
    target[1, 0, 0] = 1
    target[0, 1, 0] = 1
    target *= target
    t_2 = SparsePoly(3)
    t_2[0, 0, 1] = 2
    target *= t_2

    div = poly_div(target, t_2)

    print("target:", target)
    print("p_0:", t_2)
    print("p_x:", div)
    print("outcome:", div * t_2)

if __name__ == '__main__':
    main()