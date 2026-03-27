.data
myString: .asciz "What's up homies my names compalicious computer"

.text
.globl main
main:
    subq $8, %rsp
    xorb %al, %al
    leaq myString(%rip), %rdi
    call printf
    addq $8, %rsp
    movq $0, %rax # dosdfs
    ret
