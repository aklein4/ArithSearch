#ifndef ANNEAL_NODE_H
#define ANNEAL_NODE_H

#include <vector>
#include <string>

class AnnealNode {
    public:

        AnnealNode(int n, int* common_=nullptr, AnnealNode* on_child_=nullptr, AnnealNode* off_child_=nullptr, int size_=0, bool added_const_=false);
        ~AnnealNode();

        AnnealNode* copy();
        std::vector<AnnealNode*> get_list();
        std::string get_string();

        int n;
        int* common;
        AnnealNode* on_child;
        AnnealNode* off_child;
        int size;
        bool added_const;

};

# endif