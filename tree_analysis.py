
import math

"""
Looks at the of shapes that trees of different costs could have.

Conclusion: 
    Iterating through every tree possible tree shape becomes impossible around cost 9
    (likely lower when other combinatorics are considered),
    which is in the range of a 5th degree polynomial. Therefore, this won't work without
    peephole or other constraints/optimizations.
"""

def n(c, mem=None):
    """
    Return the number of possible tree shapes of cost c.
    """

    # memoization
    if mem == None:
        mem = {}
    if c in mem.keys():
        return mem[c]

    # recursive relationship
    answer = None
    if c == 0:
        # base case
        answer = 1
    else:
        # possible sub trees combos that could make this current tree
        answer = sum([
            n(a, mem) * n(c-1-a, mem) for a in range(0, math.floor((c-1)/2)+1)
        ])
        
    if c not in mem.keys():
        mem[c] = answer
    return answer

def main():

    print("\nCost of tree -> Number of Tree Shapes -> Number of Trees With 2 Operations \n")
    for i in range(1, 21):
        num = n(i)
        try:
            print(i, "->", num, "->", '{:.5E}'.format(2**(num)))
        except:
            print(i, "->", num, "->", "inf")
    print('')

if __name__ == "__main__":
    main()