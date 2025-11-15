#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "operations.h"

int N, M, *matI, *matP, qtdT = 0, qtdD = 0;

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

int main(int argc, char *argv[]) {
    srand(time(NULL));
    FILE *file, *out;
    
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    file = fopen(argv[1], "r");

    if (file == NULL) {
        printf("Error opening file %s\n", argv[1]);
        return 2;
    }

    fscanf(file, "%d %d", &N, &M);

    matI = (int *)malloc(N * M * sizeof(int));
    matP = (int *)malloc(N * M * sizeof(int));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            fscanf(file, "%d", &matP[INDEX(i, j, M)]);
        }
        
        if (matP[INDEX(i, j, M)] != 0) {
            qtdT++;
        }
        if (matP[INDEX(i, j, M)] == -2) {
            qtdD++;
        }
    }

    for (int i = 0; i < N*M; i++) {
        contaminateAll(i);
        healAll(i);
        removeAllDead(i);

        if (i % 2 == 0) {
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < M; y++) {
                    matI[INDEX(x, y, M)] = matP[INDEX(x, y, M)];
                }
            }
        } else {
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < M; y++) {
                    matP[INDEX(x, y, M)] = matI[INDEX(x, y, M)];
                }
            }
        }
    }

    out = fopen("infected_cpu.txt", "w");
    
    if (out == NULL) {
        printf("Error opening output file.\n");
        return 3;
    }

    fprintf(out, "Mortos: %d, Sobreviventes: %d\n", qtdD, qtdT - qtdD);
    
    fclose(file);
    fclose(out);
    return 0;
}
