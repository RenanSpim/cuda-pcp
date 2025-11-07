#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10
#define M 20

int matI[N][M];
int matP[N][M];

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

void contaminateAll(int x) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            contaminate(x % 2 == 0 ? matP : matI, i, j);
        }
    }
}

void healAll(int x) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            heal(x % 2 == 0 ? matP : matI, i, j);
        }
    }
}

void removeAllDead(int x) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            removeDead(x % 2 == 0 ? matP : matI, i, j);
        }
    }
}

int main() {
    srand(time(NULL));

    for (int i = 0; i < N*M; i++) {
        contaminateAll(i);
        healAll(i);
        removeAllDead(i);
    }
}
