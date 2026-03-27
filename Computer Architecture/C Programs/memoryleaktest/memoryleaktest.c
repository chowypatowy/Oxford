#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
// run ulimit -v 100000 in terminal first

int main(void) {
    size_t chunk_size = 1024 * 1024; // 1 MB
    int *ptr;
    unsigned long long total_allocated = 0;

    printf("Starting memory allocation...\n");

    while (1) {
        // ptr = (int *)malloc(chunk_size);
        ptr = (int *)calloc(1, chunk_size);

        if (ptr == NULL) {
            perror("malloc/calloc failed");
            printf("\nMemory limit reached!\n");
            printf("Total successfully allocated: %llu MB\n", total_allocated);
            getchar();
            break;
        }

        // Optional: Touch the memory to ensure the OS actually maps it
        ptr[0] = 0; 

        total_allocated++;
        if (total_allocated % 10 == 0) {
            printf("\rAllocated: %llu MB", total_allocated);
            fflush(stdout); // Keeps the counter updating on one line
        }
    }

    // In a real app, you'd free here, but we're about to exit anyway.
    return 0;
}
