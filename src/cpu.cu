#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10
#define M 20

int matI[N][M];
int matP[N][M];

void heal(int mat[N][M], int x, int y) {
    int val = rand() % 10000;

    if (mat[x][y] == -1)
        mat[x][y] = val < 1000 ? 1 : val < 4000 ? -1 : -2;
}

void contaminate(int mat[N][M], int x, int y) {
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

void removeDead(int mat[N][M], int x, int y) {
    if (mat[x][y] == -2)
        mat[x][y] = -3;

    if (mat[x][y] == -3)
        mat[x][y] = 0;
}

void contaminateAll(int x) {
    if (x % 2 == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                contaminate(matP, i, j);
            }
        }
    } else {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                contaminate(matI, i, j);
            }
        }
    }
}

void healAll(int x) {
    if (x % 2 == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                heal(matP, i, j);
            }
        }
    } else {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                heal(matI, i, j);
            }
        }
    }
}

void removeAllDead(int x) {
    if (x % 2 == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                removeDead(matP, i, j);
            }
        }
    } else {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                removeDead(matI, i, j);
            }
        }
    }
}

int main() {
    srand(time(NULL));

    for (int i = 0; i < N*M; i++) {
        contaminateAll(i);
        healAll(i);
        removeAllDead(i);

        if (i % 2 == 0) {
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < M; y++) {
                    matI[x][y] = matP[x][y];
                }
            }
        } else {
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < M; y++) {
                    matP[x][y] = matI[x][y];
                }
            }
        }
    }
}
