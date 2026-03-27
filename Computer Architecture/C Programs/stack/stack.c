#include <stdlib.h>
struct stack_t{
    int *base;
    int index;
    int size;
};

struct stack_t *Stack_new(){
    struct stack_t *s = malloc(sizeof(struct stack_t));
    if(s == NULL) return NULL;
    s->base = malloc(sizeof(int) * 8);
    if(s->base == NULL){
        free(s);
        return NULL;
    }
    s->index = -1;
    s->size = 8;
}

void Stack_push(struct stack_t *stack, int value, int* error){
    if(stack->index == stack->size - 1){
        int new_size = stack->size * 2;
        int* new_base = malloc(sizeof(int) * new_size);
        if(new_base == NULL){
            *error = 1;
            return;
        }
        for(int i = 0; i < stack->size; i ++){
            *(new_base + i) = *(stack->base + i);
        }
        free(stack->base);
        stack->base = new_base;
        stack->size = new_size;
        *error = 0;
    }
    stack->index += 1;
    *(stack->base + stack->index) = value;
}

int Stack_pop(struct stack_t *stack, int *error){
    if(stack->index == -1){
        *error = 2;
        return 0;
    }
    int n = *(stack->base + stack->index);
    stack->index -= 1;
    *error = 0;
    return n;

}

void Stack_delete(struct stack_t *stack){
    free(stack->base);
    free(stack);
}
