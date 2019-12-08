#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <memory.h>

// David Shane Elliott
// Quicksort Algorithm
// Based on: https://www.geeksforgeeks.org/quick-sort/

static const long Num_To_Sort = 1000000000;

void quickSortSequential(int *arr, int lowIndex, int highIndex);

void quickSortParallel(int *arr, int lowIndex, int highIndex);

int partitionArr(int *arr, int lowIndex, int highIndex);

void swapValues(int* firstValue, int* secondValue);

// Sequential version of your sort
// If you're implementing the PSRS algorithm, you may ignore this section
void sort_s(int *arr) {
    quickSortSequential(arr, 0, Num_To_Sort - 1);
}

// Parallel version of your sort
void sort_p(int *arr) {
#pragma omp parallel default(none) shared(arr)
    {
        #pragma omp single nowait
        {
            { (void) quickSortParallel(arr, 0, Num_To_Sort - 1); }
        }
    }
}

// This function is the sequential implementation
// of the quicksort algorithm
void quickSortSequential(int *arr, int lowIndex, int highIndex)
{
    if (lowIndex < highIndex)
    {
        // Get the pivot index
        int pivotIndex = partitionArr(arr, lowIndex, highIndex);

        // Recursively call function for elements
        // before and after the partition
        quickSortSequential(arr, lowIndex, pivotIndex - 1);
        quickSortSequential(arr, pivotIndex + 1, highIndex);
    }
}

// This function is the parallel implementation
// of the quicksort algorithm
void quickSortParallel(int *arr, int lowIndex, int highIndex)
{
    if (lowIndex < highIndex)
    {
        // Get the pivot index
        int pivotIndex = partitionArr(arr, lowIndex, highIndex);

        // Recursively call function for elements
        // before and after the partition
        // OpenMP pragma for threading a task
#pragma omp task default(none) firstprivate(arr, lowIndex, pivotIndex)
		{ (void)quickSortParallel(arr, lowIndex, pivotIndex - 1); }
        // OpenMP pragma for threading a task
#pragma omp task default(none) firstprivate(arr, highIndex, pivotIndex)
		{ (void)quickSortParallel(arr, pivotIndex + 1, highIndex); }
    }
}

// This function places last element, as a pivot, among the others,
// with lower values to the left, and higher values to the right
int partitionArr (int *arr, int lowIndex, int highIndex)
{
    int pivotValue = arr[highIndex];    // pivot
    int index = (lowIndex - 1); // Index variable used for swapping

    for (int j = lowIndex; j <= highIndex - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivotValue)
        {
            index++;    // increment index of smaller element
            swapValues(&arr[index], &arr[j]);
        }
    }
    swapValues(&arr[index + 1], &arr[highIndex]);
    return (index + 1);
}

// A utility function to swap two elements
void swapValues(int* firstValue, int* secondValue)
{
    int t = *firstValue;
    *firstValue = *secondValue;
    *secondValue = t;
}

int main() {
    int *arr_s = malloc(sizeof(int) * Num_To_Sort);
    long chunk_size = Num_To_Sort / omp_get_max_threads();
    // Create random number list
#pragma omp parallel num_threads(omp_get_max_threads())
    {
        int p = omp_get_thread_num();
        unsigned int seed = (unsigned int) time(NULL) + (unsigned int) p;
        long chunk_start = p * chunk_size;
        long chunk_end = chunk_start + chunk_size;
        for (long i = chunk_start; i < chunk_end; i++) {
            arr_s[i] = rand_r(&seed);
        }
    }

    // Copy the array so that the sorting function can operate on it directly.
    // Note that this doubles the memory usage.
    // You may wish to test with slightly smaller arrays if you're running out of memory.
    int *arr_p = malloc(sizeof(int) * Num_To_Sort);
    memcpy(arr_p, arr_s, sizeof(int) * Num_To_Sort);

    struct timeval start, end;

    // Get sequential runtime and print results
    printf("Timing sequential...\n");
    gettimeofday(&start, NULL);
    sort_s(arr_s);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    free(arr_s);

    // Get parallel runtime and print results
    printf("Timing parallel...\n");
    gettimeofday(&start, NULL);
    sort_p(arr_p);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    free(arr_p);

    return 0;
}

