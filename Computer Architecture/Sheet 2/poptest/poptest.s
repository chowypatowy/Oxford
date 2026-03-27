.text
.globl poptest
poptest:
    movq %rsp, %rdi
    pushq $0xABCD
    popq %rsp
    movq %rsp, %rax
    movq %rdi, %rsp
    retq
