
#include "anneal_node.h"

AnnealNode::AnnealNode(int n_, int* common_, AnnealNode* on_child_, AnnealNode* off_child_, int size_, bool added_const_) {
    n = n_;
    common = common_;
    on_child = on_child_;
    off_child = off_child_;
    size = size_;
    added_const = added_const_;
}


AnnealNode::~AnnealNode() {
    if (on_child != nullptr) {
        delete on_child;
    }
    if (off_child != nullptr) {
        delete off_child;
    }
}


AnnealNode* AnnealNode::copy() {
    AnnealNode* new_node = new AnnealNode(n);

    new_node->common = common;
    new_node->size = size;
    new_node->added_const = added_const;

    if (on_child != nullptr) {
        new_node->on_child = on_child->copy();
    }
    if (off_child != nullptr) {
        new_node->off_child = off_child->copy();
    }

    return new_node;
}


std::vector<AnnealNode*> AnnealNode::get_list() {
    std::vector<AnnealNode*> nodes;
    
    if (on_child != nullptr) {
        nodes.push_back(on_child);
    }
    if (off_child != nullptr) {
        nodes.push_back(off_child);
    }

    return nodes;
}


std::string AnnealNode::get_string() {
    std::string s;
    
    if (on_child != nullptr) {
        for (int i=0; i<n; ++i) {
            if (common[i] > 0) {
                s += "x_" + std::to_string(i) + "^" + std::to_string(common[i]);
            }
        }
        s += "(" + on_child->get_string() + ")";
    }

    if (off_child != nullptr) {
        if (s.length() > 0) {
            s += ' + ';
        }
    }

    if (added_const) {
        if (s.length() > 0) {
            s += " + ";
        }
        s += "c";
    }

    return s;
}
