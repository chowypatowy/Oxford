	.file	"main.c"
	.text
	.globl	multiply
	.type	multiply, @function
multiply:
.LFB0:
	.cfi_startproc
	endbr64
	testq	%rsi, %rsi
	jle	.L4
	movq	%rdi, %rax
	leaq	(%rdi,%rsi,8), %rcx
	movl	$1, %edx
.L3:
	imulq	(%rax), %rdx
	addq	$8, %rax
	cmpq	%rcx, %rax
	jne	.L3
.L1:
	movq	%rdx, %rax
	ret
.L4:
	movl	$1, %edx
	jmp	.L1
	.cfi_endproc
.LFE0:
	.size	multiply, .-multiply
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
