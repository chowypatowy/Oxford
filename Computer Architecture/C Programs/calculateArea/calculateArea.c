#include<stdio.h>

typedef struct {
    unsigned short width;
    unsigned short length;
    unsigned long uniqueID[2];
} Rectangle;

int calculateArea(Rectangle *rect){
    return rect->width * rect->length;
}

int main(void){
    Rectangle *rect;
    rect->width = 2;
    rect->length = 3;
    printf("%d\n", calculateArea(rect));
}
