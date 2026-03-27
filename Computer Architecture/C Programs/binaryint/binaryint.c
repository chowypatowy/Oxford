#include <stdio.h>
#include <stdlib.h>

char* binaryInt(int n){
    int N = sizeof(int) * 8;
    char *binary = (char*)malloc(32 * sizeof(char));
    // technically need 33 spots and set last one to 0 for termination
    for(int i = 0; i < N; i ++){
        int j = N - 1 - i; // Start from left of value
        binary[i] = (n & (1 << j) ? '1' : '0');
    }
    return binary;
}

int main(void){

    char* n = binaryInt(1);
    printf("%s\n", n);
    return 0;
}
