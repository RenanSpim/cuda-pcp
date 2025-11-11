#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10
#define M 20

__device__ void heal(int mat[N][M], int x, int y, unsigned int seed) {
    // Gerando numero pseudo-aleatório usando LCG (Linear Congruential Generator)
    // Como rand() não funciona no device, usamos um gerador simples
    unsigned int val = ((seed + x * M + y) * 1103515245 + 12345) % 10000;

    if (mat[x][y] == -1)
        mat[x][y] = val < 1000 ? 1 : val < 4000 ? -1 : -2;
}

__device__ void contaminate(int mat[N][M], int x, int y) {
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

__device__ void removeDead(int mat[N][M], int x, int y) {
    if (mat[x][y] == -2)
        mat[x][y] = -3;

    if (mat[x][y] == -3)
        mat[x][y] = 0;
}

__global__ void kernel(int *matI, int *matP, unsigned int seed, int num_iterations) {
    // Cada thread processa uma célula da matriz
    int tid = threadIdx.x;
    
    if (tid >= N * M) return;
    
    // Calcula posição x,y da célula que esta thread processa
    int x = tid / M;
    int y = tid % M;
    
    // Converte ponteiros para arrays 2D
    int (*mat_input)[M] = (int (*)[M])matI;
    int (*mat_output)[M] = (int (*)[M])matP;
    
    // Processa num_iterations iterações
    for (int iter = 0; iter < num_iterations; iter++) {
        // Determina qual matriz é entrada e qual é saída nesta iteração
        int (*mat_read)[M] = (iter % 2 == 0) ? mat_input : mat_output;
        int (*mat_write)[M] = (iter % 2 == 0) ? mat_output : mat_input;
        
        // Sincroniza antes de começar a iteração
        __syncthreads();
        
        // Cada thread copia sua célula
        mat_write[x][y] = mat_read[x][y];
        
        // Sincroniza após cópia
        __syncthreads();
        
        // Cada thread contamina sua célula
        contaminate(mat_write, x, y);
        
        // Sincroniza após contaminação
        __syncthreads();
        
        // Cada thread cura sua célula
        heal(mat_write, x, y, seed + iter);
        
        // Sincroniza após cura
        __syncthreads();
        
        // Cada thread remove mortos da sua célula
        removeDead(mat_write, x, y);
        
        // Sincroniza após remoção
        __syncthreads();
    }
}

int main() {
    // Criando seed de números aleatórios
    unsigned int seed = (unsigned int)time(NULL);

    // Declaração das matrizes no host
    int matI[N][M] = {0};
    int matP[N][M] = {0};
    
    // Lê matriz inicial do arquivo .txt
    FILE *file = fopen("matriz_inicial.txt", "r");
    if (file != NULL) {
        int n, m;
        // Linha 1: Lê N e M
        fscanf(file, "%d %d", &n, &m);
        
        if (n == N && m == M) {
            // Próximas N linhas: Lê M inteiros por linha
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    fscanf(file, "%d", &matP[i][j]);
                }
            }
            printf("Matriz inicial carregada de 'matriz_inicial.txt' (%dx%d)\n", n, m);
            
            // Copia para matI também
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    matI[i][j] = matP[i][j];
                }
            }
        } else {
            printf("ERRO: Dimensões do arquivo (%dx%d) não correspondem às esperadas (%dx%d)\n", n, m, N, M);
        }
        fclose(file);
    } else {
        printf("Arquivo 'matriz_inicial.txt' não encontrado. Usando matriz zerada.\n");
    }
    
    // Aloca e copia dados para o device
    int *d_matI, *d_matP;
    cudaError_t err;
    
    err = cudaMalloc(&d_matI, N * M * sizeof(int));
    if (err != cudaSuccess) {
        printf("ERRO cudaMalloc matI: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMalloc(&d_matP, N * M * sizeof(int));
    if (err != cudaSuccess) {
        printf("ERRO cudaMalloc matP: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMemcpy(d_matI, matI, N * M * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("ERRO cudaMemcpy H->D matI: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMemcpy(d_matP, matP, N * M * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("ERRO cudaMemcpy H->D matP: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Define número de iterações
    int num_iterations = N * M;  // 200 iterações
    
    // Lança o kernel com 1 bloco e N*M threads (cada thread processa uma célula)
    kernel<<<1, N * M>>>(d_matI, d_matP, seed, num_iterations);
    
    // Verifica erros no lançamento do kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERRO ao lançar kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Sincroniza
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERRO cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copia AMBAS as matrizes de volta do device para o host
    err = cudaMemcpy(matI, d_matI, N * M * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("ERRO cudaMemcpy D->H matI: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMemcpy(matP, d_matP, N * M * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("ERRO cudaMemcpy D->H matP: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Libera memória
    cudaFree(d_matI);
    cudaFree(d_matP);
    
    printf("Simulação concluída com sucesso! (%d iterações processadas)\n", num_iterations);
    
    // Determina qual matriz tem o resultado final baseado na paridade das iterações
    int (*matFinal)[M] = ((num_iterations - 1) % 2 == 0) ? matI : matP;
    
    // Contabiliza estatísticas
    int total_mortos = 0;
    int total_sobreviventes = 0;
    int total_saudaveis = 0;
    int total_infectados = 0;
    int total_vazios = 0;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int valor = matFinal[i][j];
            if (valor == 1) {
                total_saudaveis++;
                total_sobreviventes++;
            } else if (valor == -1) {
                total_infectados++;
                total_sobreviventes++;
            } else if (valor == -2 || valor == -3) {
                total_mortos++;
            } else if (valor == 0) {
                total_vazios++;
            }
        }
    }
    
    // Grava estatísticas em arquivo de saída
    FILE *output = fopen("resultado.txt", "w");
    if (output != NULL) {
        fprintf(output, "Resultado da Simulação\n");
        fprintf(output, "======================\n\n");
        fprintf(output, "Total de mortos: %d\n", total_mortos);
        fprintf(output, "Total de sobreviventes: %d\n", total_sobreviventes);
        fprintf(output, "  - Saudáveis: %d\n", total_saudaveis);
        fprintf(output, "  - Contaminados: %d\n", total_infectados);
        fprintf(output, "\nTotal de células vazias: %d\n", total_vazios);
        fprintf(output, "Total de células: %d\n", N * M);
        fclose(output);
        printf("Estatísticas salvas em 'resultado.txt'\n");
    } else {
        printf("ERRO: Não foi possível criar arquivo de saída!\n");
    }
    
    // Imprime estatísticas na tela
    printf("\nEstatísticas Finais:\n");
    printf("====================\n");
    printf("Total de mortos: %d\n", total_mortos);
    printf("Total de sobreviventes: %d\n", total_sobreviventes);
    printf("  - Saudáveis: %d\n", total_saudaveis);
    printf("  - Contaminados: %d\n", total_infectados);
    printf("Células vazias: %d\n", total_vazios);

    printf("\nMatriz Final (matP):\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("%3d ", matP[i][j]);
        }
        printf("\n");
    }

    printf("\nOutra matriz (matI):\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("%3d ", matFinal[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}