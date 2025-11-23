#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INDEX(i, j, M) ((i) * (M) + (j))

int N, M, *matI, *matP, qtdT = 0, qtdD = 0;

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
    if (mat[INDEX(x, y, M)] == -2) {
        mat[INDEX(x, y, M)] = -3;
        qtdD++;
    }

    if (mat[INDEX(x, y, M)] == -3)
        mat[INDEX(x, y, M)] = 0;
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

int main(int argc, char *argv[]) {
    srand(time(NULL));
    FILE *file, *out;
    clock_t start, end;
    double cpu_time_used;
    
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
            
            if (matP[INDEX(i, j, M)] != 0) {
                qtdT++;
            }
            if (matP[INDEX(i, j, M)] == -2) {
                qtdD++;
            }
        }
    }

    start = clock();
    
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

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    out = fopen("results/infected_cpu.txt", "w");
    
    if (out == NULL) {
        printf("Error opening output file.\n");
        return 3;
    }

    fprintf(out, "=== RESULTADOS DA SIMULACAO - CPU ===\n\n");
    fprintf(out, "Configuracao:\n");
    fprintf(out, "  Dimensoes: %d x %d\n", N, M);
    fprintf(out, "  Tempo de execucao: %.6f segundos\n\n", cpu_time_used);
    fprintf(out, "Estatisticas:\n");
    fprintf(out, "  Mortos: %d\n", qtdD);
    fprintf(out, "  Sobreviventes: %d\n", qtdT - qtdD);
    
    printf("\n=== Simulacao CPU concluida ===\n");
    printf("Tempo de execucao: %.6f segundos\n", cpu_time_used);
    
    fclose(file);
    fclose(out);
    return 0;
}

