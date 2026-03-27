long multiply(long *a, long n) {
    long i;
    long acc = 1;
    for (i = 0; i < n; i += 1) {
        acc = acc * a[i];
    }
    return acc;
}
