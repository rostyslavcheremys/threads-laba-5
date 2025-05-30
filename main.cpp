#include <iostream>
#include <ctime>
#include <omp.h>
#include <climits>

using namespace std;

constexpr int ROWS = 10000;
constexpr int COLS = 10000;

int matrix[ROWS][COLS];

struct RowMin {
    int row_index;
    long long min_sum;
};

#pragma omp declare reduction(min_row : RowMin : \
omp_out = (omp_in.min_sum < omp_out.min_sum) ? omp_in : omp_out) \
initializer(omp_priv = { -1, LLONG_MAX })

void init_matrix();

long long total_sum(int num_threads);
RowMin min_row_sum(int num_threads);

int main() {
    init_matrix();

    omp_set_nested(1);

    long long total = 0;
    RowMin min_row = {0, LLONG_MAX};

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            total = total_sum(8);
        }
        #pragma omp section
        {
            min_row = min_row_sum(8);
        }
    }

    cout << "=== RESULT ===" << endl;
    cout << "Total sum of matrix elements: " << total << endl;
    cout << "Row with the minimum sum: " << min_row.row_index << endl;
    cout << "Sum of elements in this row: " << min_row.min_sum << endl;

    return 0;
}

void init_matrix() {
    srand(time(NULL));

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            matrix[i][j] = rand();
        }
    }
}

long long total_sum(int num_threads) {
    long long sum = 0;
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum) num_threads(num_threads)

    for (int i = 0; i < ROWS; i++) {
        long long row_sum = 0;

        for (int j = 0; j < COLS; j++) {
            row_sum += matrix[i][j];
        }

        sum += row_sum;
    }

    double end = omp_get_wtime();

    cout << "Total sum calculated in " << end - start << " seconds" << " with " << num_threads << " threads." << endl;

    return sum;
}

RowMin min_row_sum(int num_threads) {
    double start = omp_get_wtime();

    RowMin global_min = {-1, LLONG_MAX};

    #pragma omp parallel for num_threads(num_threads) reduction(min_row:global_min)

    for (int i = 0; i < ROWS; i++) {
        long long row_sum = 0;

        for (int j = 0; j < COLS; j++) {
            row_sum += matrix[i][j];
        }

        if (row_sum < global_min.min_sum) {
            global_min = {i, row_sum};

            /*#pragma omp critical
            {
                if (row_sum < global_min.min_sum) {
                    global_min = {i, row_sum};
                }
            }*/
        }
    }

    double end = omp_get_wtime();
    cout << "Min row sum found in " << end - start << " seconds with " << num_threads << " threads." << endl << endl;

    return global_min;
}
