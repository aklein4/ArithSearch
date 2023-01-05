#ifndef DIVSEARCH_H
#define DIVSEARCH_H

#include "div_utils.hpp"
#include <vector>

inline float heuristic(int* a, int* d, int n) {
    int* diff = sub_vec(a, d, n);

    for (int i=0; i<n; i++) {
        if (diff[i] < 0) {
            return 0;
        }
    }

    float* root = sqrt_vec(a, n);
    float score = 0.0;

    for (int i=0; i<n; i++) {
        if (root[i] > 0) {
            score += static_cast<float>(d[i]) / root[i];
        }
    }

    return score;
}


class DivSearch {
    public:

        DivSearch(int n_, std::vector<int*> poly_, int n_rands_=0);
        ~DivSearch();

        int n;
        std::vector<int*> poly;
        int n_rands_;

        std::vector<int*> monoms;        

};


# endif