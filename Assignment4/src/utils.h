#ifndef __UTILS_H__
#define __UTILS_H__

#include <vector>
#include <random>
#include <iostream>
#include <cfloat>
using namespace std;

vector<float> get_random_vector(int n) {
    vector<float> v(n);
    for (int i = 0; i < n; i++)
        v[i] = static_cast<float>(rand()) / RAND_MAX;
    return v;
}

ostream& operator<<(ostream& os, const vector<float>& v) {
    for (size_t i = 0; i < v.size(); i++)
        os << v[i] << ' ';
    return os;
}

void print_mat(ostream& os, const vector<float>& mat, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++)
            os << mat[i*columns + j] << ' ';
        os << endl;
    }
}

void print_vec(ostream& os, const vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); i++)
        os << vec[i] << endl;
}

#endif