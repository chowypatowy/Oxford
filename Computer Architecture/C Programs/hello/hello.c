#include <stdio.h>

int main(void){
    // int numbers[] = {1, 2, 3, 4};
    // *(numbers + 3) = 2;
    // int *aPointer = numbers;
    // int *bPointer = &(numbers[0]);


    // printf("%d\n", *(numbers)); 
    // printf("%d\n", *(aPointer)); 
    // printf("%d\n", *(bPointer)); 
    //
    // printf("%zu\n", sizeof(*(numbers))); 
    // printf("%zu\n", sizeof(*(numbers + 2))); 
    //
    // printf("%p\n", &(*(numbers))); 
    // printf("%p\n", &(*(numbers + 2))); 
    //
    // printf("%p\n", (void *)&(*(numbers))); 
    // printf("%p\n", (void *)&(*(numbers + 2))); 
    

    // int numbers[] = {1, 2, 3, 4};
    // printf("%d\n", *(numbers)); 
    // int *a;
    // int b = 0;
    // a = &b;
    // *a += 1;
    // printf("%d\n", b); 
    
    int P[5];
    short Q[2];
    char* R[9];
    double *S[10];
    short *T[2];

    char* pointer;
    char c = 'X';
    pointer = &c;
    R[0] = pointer;

    printf("%ld\n", sizeof(P));
    printf("%ld\n", sizeof(Q));
    printf("%ld\n", sizeof(R));
    printf("%ld\n", sizeof(S));
    printf("%ld\n", sizeof(T));

    printf("%c\n", **R);
    return 0;

}
