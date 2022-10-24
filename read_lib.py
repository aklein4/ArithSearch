
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
                        self.library.append({'p': row[1], 'leaf': True, "val": None, "op": None, "op_a": None, "op_b": None, "cost": int(row[5]), "order": int(row[6])})
                    else:
                        # constant
                        self.library.append({'p': row[1], 'leaf': True, "val": int(row[1]), "op": None, "op_a": None, "op_b": None, "cost": int(row[5]), "order": int(row[6])})
                else:
                    # regular node
                    self.library.append({
                        'p': row[1], 'leaf': False, "val": None,
                        "op": OPERATIONS.ADD if row[2]=="ADD" else OPERATIONS.MULT,
                        "op_a": int(row[3]), "op_b": int(row[4]), "cost": int(row[5]), "order": int(row[6])
                    })
    

    def show_data(self):
        orders = []
        costs = []
        for d in self.library:
            orders.append(d["order"])
            costs.append(d["cost"])
        plt.scatter(orders, costs)
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


def main(filename):
    # read in file
    trees = LibReader(filename)

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
    main(sys.argv[1])