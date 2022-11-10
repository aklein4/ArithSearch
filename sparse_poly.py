
import numpy as np
from sklearn.metrics import mean_squared_error

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
            return new_p

        else:
            raise ValueError("invalid type in __add__")


    def __iadd__(self, other):
        return self + other


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


def poly_MSE(p1: SparsePoly, p2: SparsePoly):
    all_keys = set(p1.dict.keys()) | set(p2.dict.keys())

    l1 = []
    l2 = []
    for k in all_keys:
        l1.append(p1.dict.get(k, 0))
        l2.append(p2.dict.get(k, 0))

    return mean_squared_error(np.array(l1), np.array(l2))


def main():
    s = SparsePoly(2)
    s[1, 1] = 1
    s[0, 2] = 1
    s_2 = s.copy()
    s *= s
    print(poly_MSE(s, s_2))

if __name__ == '__main__':
    main()