#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

int N, M;
int qtdT = 0, qtdD = 0;

static uint32_t rng_state = 123456789;

static inline uint32_t fast_rand() {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rng_state = x;
}

int main(int argc, char *argv[]) {
    FILE *file, *out;
    clock_t start, end;
    double cpu_time_used;
    int *mat;

    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("Error opening file %s\n", argv[1]);
        return 2;
    }

    if (fscanf(file, "%d %d", &N, &M) != 2) {
        printf("Erro ao ler dimensoes.\n");
        return 4;
    }

    mat = (int *)malloc(N * M * sizeof(int));

    for (int i = 0; i < N * M; i++) {
        fscanf(file, "%d", &mat[i]);
        if (mat[i] != 0) qtdT++;
        if (mat[i] == -2) qtdD++;
    }
    fclose(file);

    rng_state = (uint32_t)time(NULL);

    start = clock();

    int total_cells = N * M;
    int generations = total_cells;

    for (int gen = 0; gen < generations; gen++) {
        
        int *ptr = mat; 
        
        for (int i = 0; i < N; i++) {
            int *row_up   = (i > 0)     ? (ptr - M) : ptr;
            int *row_down = (i < N - 1) ? (ptr + M) : ptr;

            for (int j = 0; j < M; j++) {
                int current_val = *ptr;

                if (current_val == 1) {
                    int infected = 0;
                    
                    if (row_up[j] < 0) infected = 1;
                    else if (row_down[j] < 0) infected = 1;
                    else if (j > 0 && ptr[-1] < 0) infected = 1;
                    else if (j < M - 1 && ptr[1] < 0) infected = 1;

                    if (infected) {
                        *ptr = -1;
                        current_val = -1;
                    }
                }

                if (current_val == -1) {
                    int val = fast_rand() % 10000;
                    
                    if (val < 1000) *ptr = 1;
                    else if (val < 4000) *ptr = -1;
                    else *ptr = -2;
                    
                    current_val = *ptr;
                }

                if (current_val == -2) {
                    *ptr = -3;
                    qtdD++;
                } else if (current_val == -3) {
                    *ptr = 0;
                }

                ptr++;
            }
        }
    }

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    out = fopen("results/infected_cpu.txt", "w");
    if (out == NULL) {
        out = fopen("infected_cpu.txt", "w");
    }

    if (out) {
        fprintf(out, "=== RESULTADOS DA SIMULACAO - CPU OTIMIZADO ===\n\n");
        fprintf(out, "Configuracao:\n");
        fprintf(out, "  Dimensoes: %d x %d\n", N, M);
        fprintf(out, "  Tempo de execucao: %.6f segundos\n\n", cpu_time_used);
        fprintf(out, "Estatisticas:\n");
        fprintf(out, "  Mortos: %d\n", qtdD);
        fprintf(out, "  Sobreviventes: %d\n", qtdT - qtdD);
        fclose(out);
    } else {
        printf("Erro ao criar arquivo de saida.\n");
    }
    
    printf("\n=== Simulacao CPU OTIMIZADA concluida ===\n");
    printf("Tempo de execucao: %.6f segundos\n", cpu_time_used);
    
    free(mat);
    return 0;
}