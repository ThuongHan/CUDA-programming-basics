#include "utils.h"
#include <chrono>
using namespace std::chrono;

float square_length(const vector<float>& v) {
    float sum = 0;
    for (size_t i = 0; i < v.size(); i++)
        sum += v[i] * v[i];
    return sum;
}

int main() {
    //srand(time(nullptr));
    const int n = 1000000;

    auto start = high_resolution_clock::now();  // Timer start

    vector<float> h_v = get_random_vector(n);
    
    cout << "v: " << h_v << endl;

    float length = sqrt(square_length(h_v));
    cout << "length: " << length << endl;
    for (int i = 0; i < n; i++)
        h_v[i] /= length;
    length = sqrt(square_length(h_v));

    auto stop = high_resolution_clock::now();   // Timer end

    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "length after normalization: " << length << endl;
    cout << "Execution time: " << duration.count() << " milliseconds" << endl;

    return 0;
}

// Execution time: 4597 milliseconds
