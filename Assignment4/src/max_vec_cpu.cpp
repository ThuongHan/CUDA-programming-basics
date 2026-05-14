#include "utils.h"
#include <tuple>
#include <chrono>
using namespace std::chrono;

tuple<float, int> max_element(const vector<float>& v) {
    float max_el = -FLT_MAX;
    int max_idx = 0;
    for (size_t i = 0; i < v.size(); i++)
        if (v[i] > max_el) {
            max_el = v[i];
            max_idx = i;
        }
    return tuple<float, int>(max_el, max_idx);
}

int main() {
    //srand(time(nullptr));
    const int n = 1000000;

    auto start = high_resolution_clock::now();

    vector<float> h_v = get_random_vector(n);
    
    cout << "v: " << h_v << endl;

    auto [max_val, max_idx] = max_element(h_v);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "max element: " << max_val << " at index: " << max_idx << endl;
    cout << "Execution time " << duration.count() << " milliseconds" << endl;
}