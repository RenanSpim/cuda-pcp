#include <stdlib.h>

#define INDEX(i, j, M) ((i) * (M) + (j))

void heal(int *mat, int x, int y) {
    int val = rand() % 10000;

    if (mat[INDEX(x, y, M)] == -1)
        mat[INDEX(x, y, M)] = val < 1000 ? 1 : val < 4000 ? -1 : -2;
}

void contaminate(int *mat, int x, int y) {
    if (mat[INDEX(x, y, M)] != 1) return;

    if (
        (x > 0       && mat[(x-1) * M + y] < 0) ||
        (x < (N - 1) && mat[(x+1) * M + y] < 0) ||
        (y > 0       && mat[x * M + (y-1)] < 0) ||
        (y < (M - 1) && mat[x * M + (y+1)] < 0)
    ) {
        mat[INDEX(x, y, M)] = -1;
    }
}

void removeDead(int *mat, int x, int y) {
    if (mat[INDEX(x, y, M)] == -2)
        mat[INDEX(x, y, M)] = -3;

    if (mat[INDEX(x, y, M)] == -3)
        mat[INDEX(x, y, M)] = 0;
}
