
from circuit import OPERATIONS, Node, LeafNode, treequals

def main():

    x = LeafNode()
    c = LeafNode(1)
    op = Node(OPERATIONS.ADD, x, c)
    op_2 = Node(OPERATIONS.ADD, c, x)

    root = Node(OPERATIONS.ADD, op, op)
    root_2 = Node(OPERATIONS.ADD, op, op_2)

    print(treequals(root, root_2))

if __name__ == '__main__':
    main()