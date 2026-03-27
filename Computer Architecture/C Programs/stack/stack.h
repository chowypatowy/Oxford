struct stack_t;
struct stack_t *Stack_new();
void Stack_push(struct stack_t *stack, int value, int *error);
int Stack_pop(struct stack_t *stack, int *error);
void Stack_delete(struct stack_t *stack);
