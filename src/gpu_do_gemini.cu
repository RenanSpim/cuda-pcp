#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// --- Definições do Problema [cite: 5, 6, 7, 8] ---
#define HEALTHY 1
#define SICK -1
#define DEAD -2
#define EMPTY 0

// --- Probabilidades [cite: 38, 39, 40] ---
#define PROB_CURE 999     // 0-999 (1000 valores = 10%)
#define PROB_STAY_SICK 3999 // 1000-3999 (3000 valores = 30%)
// 4000+ (6000 valores = 60%) é a morte

/**
 * Macro para verificação de erros CUDA.
 * Facilita o debug ao reportar falhas em chamadas da API CUDA.
 */
#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t err_ = (err);                                            \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/**
 * Kernel CUDA: Inicializa os estados do gerador de números aleatórios (cuRAND).
 * Cada thread da GPU precisa do seu próprio estado para gerar números
 * aleatórios independentes e de forma paralela.
 */
__global__ void init_curand_states(curandState *states, unsigned long seed, int N, int M) {
    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * M + (blockIdx.x * blockDim.x + threadIdx.x);
    int total_threads = N * M;

    if (idx >= total_threads) return;

    // Inicializa o estado de cada thread com uma semente e sequência únicas 
    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * Kernel CUDA: Executa um passo (iteração) da simulação. [cite: 14]
 * Lê de grid_in e escreve o novo estado em grid_out.
 */
__global__ void simulate_step(const int *grid_in, int *grid_out, 
                             curandState *states, int *d_sick_count, 
                             int *d_total_deaths, int N, int M) 
{
    // Calcula o índice 1D a partir do ID da thread 2D
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * M + col;

    // Garante que a thread está dentro dos limites da matriz
    if (row >= N || col >= M) return;

    int current_state = grid_in[idx];
    int new_state = current_state; // Por padrão, o estado não muda

    switch (current_state) {
        case HEALTHY: { // Pessoa saudável [cite: 6]
            bool infected = false;
            // Verifica vizinhança horizontal e vertical [cite: 9]
            // Norte
            if (row > 0 && (grid_in[idx - M] == SICK || grid_in[idx - M] == DEAD))
                infected = true;
            // Sul
            if (row < N - 1 && (grid_in[idx + M] == SICK || grid_in[idx + M] == DEAD))
                infected = true;
            // Oeste
            if (col > 0 && (grid_in[idx - 1] == SICK || grid_in[idx - 1] == DEAD))
                infected = true;
            // Leste
            if (col < M - 1 && (grid_in[idx + 1] == SICK || grid_in[idx + 1] == DEAD))
                infected = true;

            if (infected) {
                new_state = SICK; // Torna-se contaminada [cite: 5]
            }
            break;
        }

        case SICK: { // Pessoa contaminada [cite: 5]
            // Pega o estado aleatório desta thread
            curandState *local_state = &states[idx];
            // Gera número aleatório 0-9999 
            int r = curand(local_state) % 10000;

            if (r <= PROB_CURE) { // Cura [cite: 38]
                new_state = HEALTHY;
            } else if (r <= PROB_STAY_SICK) { // Continua doente [cite: 39]
                new_state = SICK;
            } else { // Morre [cite: 40]
                new_state = DEAD;
                // Incrementa o contador total de mortes (operação atômica)
                atomicAdd(d_total_deaths, 1);
            }
            break;
        }

        case DEAD: { // Pessoa morta [cite: 7]
            // Desaparece do mapa na próxima iteração [cite: 11]
            new_state = EMPTY;
            break;
        }

        case EMPTY: // Ponto vazio [cite: 8]
            // Permanece vazio
            new_state = EMPTY;
            break;
    }

    // Escreve o resultado na matriz de saída
    grid_out[idx] = new_state;

    // Se o novo estado é "contaminado", incrementa o contador de doentes
    // para a próxima iteração (usado para a condição de parada) [cite: 15]
    if (new_state == SICK) {
        atomicAdd(d_sick_count, 1);
    }
}

/**
 * Função Host (CPU): Lê o arquivo de entrada.
 */
int* read_input_file(const char *filename, int *N, int *M) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Erro ao abrir arquivo de entrada");
        exit(EXIT_FAILURE);
    }

    // Lê N e M 
    fscanf(f, "%d %d", N, M);
    if (*N <= 0 || *M <= 0) {
        fprintf(stderr, "Dimensões N e M inválidas.\n");
        exit(EXIT_FAILURE);
    }

    int *grid = (int *)malloc((*N) * (*M) * sizeof(int));
    if (!grid) {
        fprintf(stderr, "Falha ao alocar memória do host.\n");
        exit(EXIT_FAILURE);
    }

    // Lê o restante da matriz 
    for (int i = 0; i < *N; i++) {
        for (int j = 0; j < *M; j++) {
            fscanf(f, "%d", &grid[i * (*M) + j]);
        }
    }

    fclose(f);
    return grid;
}

/**
 * Função Host (CPU): Escreve o arquivo de saída.
 */
void write_output_file(const char *filename, int total_survivors, int total_deaths) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("Erro ao abrir arquivo de saída");
        exit(EXIT_FAILURE);
    }

    // Escreve os totais 
    fprintf(f, "Sobreviventes: %d\n", total_survivors);
    fprintf(f, "Mortos: %d\n", total_deaths);
    fclose(f);
    printf("Simulação concluída. Resultados salvos em '%s'.\n", filename);
    printf("Sobreviventes: %d\nMortos: %d\n", total_survivors, total_deaths);
}

/**
 * Função Host (CPU): Conta os sobreviventes na grade final.
 */
int count_survivors(const int *grid, int N, int M) {
    int survivors = 0;
    int total_cells = N * M;
    for (int i = 0; i < total_cells; i++) {
        // Sobreviventes = saudáveis + contaminados 
        if (grid[i] == HEALTHY || grid[i] == SICK) {
            survivors++;
        }
    }
    return survivors;
}


// --- Função Principal ---
int main() {
    int N, M;
    int *h_grid; // Matriz no Host (CPU)

    // 1. Leitura de dados [cite: 17]
    h_grid = read_input_file("matriz_inicial.txt", &N, &M);
    int total_cells = N * M;
    size_t grid_size = total_cells * sizeof(int);

    // 2. Alocação de Memória na GPU (Device)
    int *d_grid_a, *d_grid_b; // Buffers para ping-pong (leitura/escrita)
    curandState *d_states;      // Estados do cuRAND
    int *d_sick_count;          // Contador de doentes (condição de parada)
    int *d_total_deaths;        // Contador total de mortes

    CUDA_CHECK(cudaMalloc((void **)&d_grid_a, grid_size));
    CUDA_CHECK(cudaMalloc((void **)&d_grid_b, grid_size));
    CUDA_CHECK(cudaMalloc((void **)&d_states, total_cells * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc((void **)&d_sick_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_total_deaths, sizeof(int)));

    // 3. Inicialização
    // Copia matriz inicial do Host (CPU) para o Device (GPU)
    CUDA_CHECK(cudaMemcpy(d_grid_a, h_grid, grid_size, cudaMemcpyHostToDevice));
    // Zera os contadores na GPU
    CUDA_CHECK(cudaMemset(d_sick_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_total_deaths, 0, sizeof(int)));

    // Define a configuração de grid e blocos
    // (Para os testes [cite: 25-32], você mudará estes valores)
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Lança o kernel para inicializar o cuRAND
    init_curand_states<<<numBlocks, threadsPerBlock>>>(d_states, (unsigned long)time(NULL), N, M);
    CUDA_CHECK(cudaGetLastError());

    // 4. Loop da Simulação
    int *d_in = d_grid_a;  // Ponteiro para a grade de entrada da iteração
    int *d_out = d_grid_b; // Ponteiro para a grade de saída da iteração
    int h_sick_count = 0;  // Contador de doentes no Host
    int max_iterations = N * M; // Condição de parada [cite: 15]

    printf("Iniciando simulação %dx%d para %d iterações...\n", N, M, max_iterations);

    for (int iter = 0; iter < max_iterations; iter++) {
        // Zera o contador de doentes *antes* de executar o kernel
        CUDA_CHECK(cudaMemset(d_sick_count, 0, sizeof(int)));

        // Lança o kernel da simulação
        simulate_step<<<numBlocks, threadsPerBlock>>>(
            d_in, d_out, d_states, d_sick_count, d_total_deaths, N, M
        );
        CUDA_CHECK(cudaGetLastError());

        // Sincroniza para garantir que o kernel terminou
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copia o novo contador de doentes da GPU para a CPU
        CUDA_CHECK(cudaMemcpy(&h_sick_count, d_sick_count, sizeof(int), cudaMemcpyDeviceToHost));

        // Condição de parada: Ninguém mais está contaminado [cite: 15]
        if (h_sick_count == 0) {
            printf("Simulação parada na iteração %d (sem contaminados).\n", iter + 1);
            // Troca os ponteiros mais uma vez para que d_in aponte para o último
            // resultado válido (que está em d_out)
            int *temp = d_in;
            d_in = d_out;
            d_out = temp;
            break;
        }

        // Troca os ponteiros (ping-pong):
        // A saída (d_out) desta iteração será a entrada (d_in) da próxima
        int *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    // 5. Coleta de Resultados
    int h_total_deaths = 0;
    // Copia o total de mortes da GPU para a CPU
    CUDA_CHECK(cudaMemcpy(&h_total_deaths, d_total_deaths, sizeof(int), cudaMemcpyDeviceToHost));
    // Copia a grade final (que está em d_in) da GPU para a CPU
    CUDA_CHECK(cudaMemcpy(h_grid, d_in, grid_size, cudaMemcpyDeviceToHost));

    // Conta os sobreviventes na matriz final (no host)
    int total_survivors = count_survivors(h_grid, N, M);

    // 6. Saída de dados [cite: 19]
    write_output_file("saida.txt", total_survivors, h_total_deaths);

    // 7. Limpeza da Memória
    free(h_grid);
    CUDA_CHECK(cudaFree(d_grid_a));
    CUDA_CHECK(cudaFree(d_grid_b));
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_sick_count));
    CUDA_CHECK(cudaFree(d_total_deaths));

    return 0;
}