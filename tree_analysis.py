
import math

def n(c, mem=None):
    """
    Return the number of possible tree shapes of cost c.
    """
    if mem == None:
        mem = {}
    if c in mem.keys():
        return mem[c]

    answer = None
    if c == 0:
        answer = 1
    else:
        answer = sum([
            n(a, mem) * n(c-1-a, mem) for a in range(0, math.floor((c-1)/2)+1)
        ])
        
    if c not in mem.keys():
        mem[c] = answer
    return answer

def main():
    print("\nCost of tree -> Number of Trees -> Number of Trees (With 2 Operations) \n")
    for i in range(20):
        try:
            print(i, "->", n(i), "->", '{:.5E}'.format(2**(n(i))))
        except:
            print(i, "->", n(i), "->", "inf")
    print('')

if __name__ == "__main__":
    main()