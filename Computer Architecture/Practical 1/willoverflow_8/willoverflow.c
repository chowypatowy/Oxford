#include <stdio.h>
#include <stdint.h>

int willOverflow1(uint8_t a, uint8_t b) {
    return (a + b < a);
}

int willOverflow2(uint8_t a, uint8_t b) {
    uint8_t c = a + b;
    return (c < a);
}

int main(){
    uint8_t a = 128;
    uint8_t b = 128;
    printf("%d\n", willOverflow1(a, b));
    printf("%d\n", willOverflow2(a, b));

}
