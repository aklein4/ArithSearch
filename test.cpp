
#include "div_utils.hpp"
#include <iostream>

int main() {
    int* a = new int[5];
    int* b = new int[5];

    for (int i=0; i<5; ++i) {
        a[i] = i*i;
        b[i] = i*i*i;
    }

    std::cout << "a:" << std::endl;
    for (int i=0; i<5; ++i) {
        std::cout << a[i] << std::endl;
    }

    std::cout << "b:" << std::endl;
    for (int i=0; i<5; ++i) {
        std::cout << b[i] << std::endl;
    }

    int* c = add_vec(a, b, 5);
    std::cout << "c:" << std::endl;
    for (int i=0; i<5; ++i) {
        std::cout << c[i] << std::endl;
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}