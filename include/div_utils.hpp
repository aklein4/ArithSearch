#ifndef DIV_UTILS_H
#define DIV_UTILS_H

#include <cmath>

int* add_vec(int* a, int* b, int n) {
    int* c = new int[n];
    for(int i=0; i<n; ++i) {
        c[i] = a[i] + b[i];
    }
    return c;
}

int* sub_vec(int* a, int* b, int n) {
    int* c = new int[n];
    for(int i=0; i<n; ++i) {
        c[i] = a[i] - b[i];
    }
    return c;
}

float* sqrt_vec(int* a, int n) {
    float* c = new float[n];
    for(int i=0; i<n; ++i) {
        c[i] = sqrt(static_cast<float>(a[i]));
    }
    return c;
}

# endif