#include <stdio.h>
#include "stack.h"

int main(void){
    struct stack_t *stack;
    stack = Stack_new();
    int error = 0;
    Stack_push(stack, 4, &error);
    Stack_push(stack, 4, &error);
    Stack_push(stack, 3, &error);
    printf("%d\n", Stack_pop(stack, &error));
    printf("%d\n", Stack_pop(stack, &error));
    Stack_push(stack, 3, &error);
    Stack_push(stack, 1, &error);
    printf("%d\n", Stack_pop(stack, &error));
    Stack_push(stack, 22, &error);
    printf("%d\n", Stack_pop(stack, &error));
    printf("%d\n", Stack_pop(stack, &error));
    printf("%d\n", Stack_pop(stack, &error));
    printf("%d\n", Stack_pop(stack, &error));
    printf("%d\n", Stack_pop(stack, &error));
    printf("%d\n", Stack_pop(stack, &error));

}
