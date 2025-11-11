#include <stdlib.h>

void heal(int **mat, int x, int y) {
    int val = rand() % 10000;

    if (mat[x][y] == -1)
        mat[x][y] = val < 1000 ? 1 : val < 4000 ? -1 : -2;
}

void contaminate(int **mat, int x, int y) {
    if (mat[x][y] != 1) return;

    if (
        (x > 0       && mat[x-1][y] < 0) ||
        (x < (N - 1) && mat[x+1][y] < 0) ||
        (y > 0       && mat[x][y-1] < 0) ||
        (y < (M - 1) && mat[x][y+1] < 0)
    ) {
        mat[x][y] = -1;
    }
}

void removeDead(int **mat, int x, int y) {
    if (mat[x][y] == -2)
        mat[x][y] = -3;

    if (mat[x][y] == -3)
        mat[x][y] = 0;
}
