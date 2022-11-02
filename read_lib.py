
from circuit import OPERATIONS, Node, LeafNode
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt

class LibReader:

    def __init__(self, filename):
        """
        :param filename: name of file to load
        """
        self.library = [] # contains dicts about the library for per-access reconstruction

        # read file
        with open(filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, dialect='excel')

            header = True
            for row in spamreader:
                # skip header
                if header:
                    header = False
                    continue

                if row[2] == "":
                    # leaf node
                    if row[1] == 'x':
                        # argument
                        self.library.append({
                            'p': row[1], 'leaf': True, "val": None,
                            "op": None, "op_a": None, "op_b": None, "cost": int(row[5]), "order": int(row[6]), "depth": int(row[7]) if len(row)>7 else 0
                        })
                    else:
                        # constant
                        self.library.append({
                            'p': row[1], 'leaf': True, "val": int(row[1]),
                            "op": None, "op_a": None, "op_b": None, "cost": int(row[5]), "order": int(row[6]), "depth": int(row[7]) if len(row)>7 else 0
                        })
                else:
                    # regular node
                    self.library.append({
                        'p': row[1], 'leaf': False, "val": None,
                        "op": OPERATIONS.ADD if row[2]=="ADD" else OPERATIONS.MULT,
                        "op_a": int(row[3]), "op_b": int(row[4]), "cost": int(row[5]), "order": int(row[6]), "depth": int(row[7]) if len(row)>7 else 0
                    })
    

    def show_data(self):
        """
        Plot the order of polynomials vs their minimum found cost.
        """
        orders = []
        costs = []
        depths = []
        for d in self.library:
            orders.append(d["order"])
            costs.append(d["cost"])
            depths.append(d["depth"])

        #plt.scatter(orders, costs)
        #plt.show()
        #plt.clf()
        vals, bins, cont = plt.hist(depths, bins=[x-0.5 for x in range(0, 21)])
        plt.clf()
        plt.plot(vals)
        # tot_combs = 0
        # taken_combs = 0
        # for i in range(0, len(vals)):
        #     for j in range(0, i+1):
        #         tot_combs += vals[i]*vals[j]
        #         if i+j+1 <= 14:
        #             taken_combs += vals[i]*vals[j]
        # print("total:", tot_combs, "- taken:", taken_combs, "- perc:", taken_combs/tot_combs)
        plt.show()


    def __getitem__(self, item):
        """
        Recursively reconstruct and return the Node tree at the given index
        :param item: Index of tree to get
        :return Tree at item
        """
        # get describing dict
        d = self.library[item]

        if d["leaf"]:
            # return leaf
            return LeafNode(d["val"]), d['p']
        
        # return recursive construction from children
        return Node(d["op"], self[d["op_a"]][0], self[d["op_b"]][0]), d['p']


def compare_files(file_1, file_2):

    # read file 1
    print("reading file 1...")
    contents_1 = {}
    depths_1 = {}
    with open(file_1, newline='') as csvfile:
        spamreader = csv.reader(csvfile, dialect='excel')

        header = True
        for row in spamreader:
            # skip header
            if header:
                header = False
                continue
            
            # log cost
            contents_1[row[1]] = int(row[5])
            if len(row) > 7:
                depth = int(row[7])
                depths_1[row[1]] = int(row[7])

    # read file 2
    print("reading file 2...")
    contents_2 = {}
    depths_2 = {}
    with open(file_2, newline='') as csvfile:
        spamreader = csv.reader(csvfile, dialect='excel')

        header = True
        for row in spamreader:
            # skip header
            if header:
                header = False
                continue

            # log cost
            contents_2[row[1]] = int(row[5])
            if len(row) > 7:
                depths_2[row[1]] = int(row[7])

    if len(depths_1.keys()) == 0 or len(depths_2.keys()) == 0:
        print("Warning: no depths found")

    # compare the polynomials
    one_better = 0
    two_better = 0
    shared_nomials = 0
    example_poly = None
    for key in contents_1.keys():
        if key in contents_2.keys():
            shared_nomials += 1
            if contents_1[key] < contents_2[key]:
                one_better += 1
                if example_poly == None:
                    example_poly = key
            elif contents_1[key] > contents_2[key]:
                two_better += 1
                if example_poly == None:
                    example_poly = key
    print("SHARED POLYNOMIALS:", shared_nomials)
    print("FILE ONE COST LESS:", one_better)
    print("FILE TWO COST LESS:", two_better)

    if example_poly != None:
        print(" - Example with Different Costs:", example_poly)

    one_depth_better = 0
    two_depth_better = 0
    example_poly = None
    tester = None
    for key in depths_1.keys():
        if key in depths_2.keys():
            if depths_1[key] < depths_2[key]:
                one_depth_better += 1
                if example_poly == None:
                    example_poly = key
                    tester = [depths_1[key], depths_2[key]]
            elif depths_1[key] > depths_2[key]:
                two_depth_better += 1
                if example_poly == None:
                    example_poly = key
                    tester = [depths_1[key], depths_2[key]]
    print("FILE ONE DEPTH LESS:", one_depth_better)
    print("FILE TWO DEPTH LESS:", two_depth_better)

    if example_poly != None:
        print(" - Example with Different Depths:", example_poly)

def main(filename):
    # read in file
    print("reading file... ")
    trees = LibReader(filename)
    print("done.")

    trees.show_data()

    # print interactively
    while True:

        # get input index
        com = input("Index (enter to quit): ")
        if com == "":
            print(' ')
            exit()

        # print polynomial and tree
        tree, p = trees[int(com)]
        print('\n'+p+'\n')
        tree.show()
        print('')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError("Please enter filename as argument.")
    elif len(sys.argv) == 3:
        compare_files(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1])