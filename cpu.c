#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10
#define M 20

int mat[N][M];

void heal(int x, int y) {
    int val = rand() % 10000;

    if (mat[x][y] == -1)
        mat[x][y] = val < 1000 ? 1 : val < 4000 ? -1 : -2;
}

void contaminate(int x, int y) {
    if (mat[x][y] != 1) return;

    if (
        (x > 0 && mat[x-1][y] < 0)       ||
        (y > 0 && mat[x][y-1] < 0)       ||
        (x < (N - 1) && mat[x+1][y] < 0) ||
        (y < (M - 1) && mat[x][y+1] < 0)
    ) {
        mat[x][y] = -1;
    }
}

void removeDead(int x, int y) {
    if (mat[x][y] == -2)
        mat[x][y] = -3;

    if (mat[x][y] == -3)
        mat[x][y] = 0;
}

void contaminateAll() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            contaminate(i, j);
        }
    }
}

void healAll() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            heal(i, j);
        }
    }
}

void removeAllDead() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            removeDead(i, j);
        }
    }
}

int main() {
    srand(time(NULL));

    for (int i = 0; i < N*M; i++) {
        contaminateAll();
        healAll();
        removeAllDead();
    }
}
