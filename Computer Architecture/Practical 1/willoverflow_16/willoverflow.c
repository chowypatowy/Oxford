#include <stdio.h>
#include <stdint.h>

int willOverflow1(uint16_t a, uint16_t b) {
    return (a + b < a);
}

int willOverflow2(uint16_t a, uint16_t b) {
    uint16_t c = a + b;
    return (c < a);
}

int main(){
    uint16_t a = 32768;
    uint16_t b = 32768;
    printf("%d\n", willOverflow1(a, b));
    printf("%d\n", willOverflow2(a, b));

}
