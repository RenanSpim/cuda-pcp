#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t err_ = (err);                                            \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel para inicializar os estados do cuRAND
__global__ void setup_curand(curandState *state, unsigned long seed, int N, int M) {
    int tid = threadIdx.x;
    if (tid < N * M) {
        // Inicializa o gerador para cada thread com seed única
        curand_init(seed, tid, 0, &state[tid]);
    }
}

__global__ void kernel(int *matP, int *matI, int *deaths, int *survivors, int N, int M, int max_iter, curandState *states){
    
    printf("Criando Threads\n");
    // Criando Threads
    int tid = threadIdx.x;
    if(tid >= N*M) return;

    // Estado local do gerador aleatório
    curandState localState = states[tid];

    // Ponteiros para matriz de entrada e saída
    int *matIn = matP;
    int *matOut = matI;

    printf("Antes do laco for\n");
    for(int i=0; i<max_iter; i++){
        
        // Determina a matriz de entrada e saída para esta iteração
        int partiy  = i % 2;
        if (partiy == 0) {
            matIn = matP;
            matOut = matI;
        } else {
            matIn = matI;
            matOut = matP;
        }

        // Garante que todas as threads definiram seus ponteiros
        __syncthreads(); 

        // Atribui o valor atual da célula
        int in_val = matIn[tid];
        int out_val = in_val;

        // Contaminate
        if (in_val == 1) { // Saudável
            if (
                (tid%M > 0    && matIn[tid-1] < 0) || // Vizinho esquerdo
                (tid%M < M-1  && matIn[tid+1] < 0) || // Vizinho direito
                (tid/M < N-1  && matIn[tid+M] < 0) || // Vizinho baixo
                (tid/M > 0    && matIn[tid-M] < 0)    // Vizinho cima
            ){
                out_val = -1; // Contamina
            }
        } // Heal
        else if (in_val == -1) { // Infectado
            unsigned int chance = curand(&localState)%10000;
            if (chance < 1000)
                out_val = 1;     // Fica saudável
            else if (chance < 4000)
                out_val = -1;    // Continua infectado
            else
                out_val = -2;    // Morre
        } // RemoveDead
        else if (in_val == -2) { // Morto (primeira iteracao
            out_val = -3;
            atomicAdd(deaths, 1);
        } 
        else if (in_val == -3) { // Morto (segunda iteração)
            out_val = 0;
        }
        
        // Espera todas as threads sincronizarem após a leitura da matIn
        __syncthreads();
        
        // Cada thread escreve na matOut
        matOut[tid] = out_val;
        
        // Espera todas as threads sincronizarem após escrever na matOut
        __syncthreads();
    }
    // sincronizando a última iteração para todas as threads
    __syncthreads();

    // Contagem de sobrevivente (infectados e saudáveis)
    if (matOut[tid] != 0 && matOut[tid] > -2) {
        atomicAdd(survivors, 1);
    }
    
    // Salva o estado atualizado de volta
    states[tid] = localState;

    // Sincroniza antes de sair
    __syncthreads();
}

int main(void){
    
    // Declarando variaveis de dimensoes
    int N, M;
    int *h_survivors = (int*)malloc(sizeof(int));
    int *h_deaths = (int *)malloc(sizeof(int));
    *h_survivors = 7;
    *h_deaths = 7;

    // Abrindo arquivo da matriz de entrada
    FILE *fileInput = fopen("/home/mario/computaria/2sem-2025/pcp/cuda-pcp/src/matriz_inicial.txt", "r");
    if(fileInput == NULL){
        printf("Erro ao abrir o arquivo de entrada.\n");
        return 1;
    }

    // Atribuindo as dimensões da matriz
    fscanf(fileInput, "%d", &N);
    fscanf(fileInput, "%d", &M);

    // Declarando matrizes no host
    int *h_matP = (int*)malloc(N*M*sizeof(int));

    // Atribuindo valores as matrizes do host
    for(int i=0;i<N*M;i++){
        fscanf(fileInput, "%d", &h_matP[i]);
    }
    fclose(fileInput);  ;

    // Printando a matriz inicial (DEBUG)
    for(int i=0;i<N*M;i++){
        printf("\t[%d] ", h_matP[i]);
        if((i+1)%M==0)
            printf("\n");
    }
    // ----------------------------------------------------;

    // Declarando variaveis no device e alocando memória
    int *d_matI, *d_matP, *d_deaths, *d_survivors;
    curandState *d_states;
    cudaError_t err;

    err = cudaMalloc(&d_matI, N*M*sizeof(int));
    if(err != cudaSuccess){
        printf("Erro de alocacao de memoria para d_matI: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_matP, N*M*sizeof(int));
    if(err != cudaSuccess){
        printf("Erro de alocacao de memoria para d_matP: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_deaths, sizeof(int));
    if(err != cudaSuccess){
        printf("Erro de alocacao de memoria para d_matP: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_survivors, sizeof(int));
    if(err != cudaSuccess){
        printf("Erro de alocacao de memoria para d_matP: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Aloca memória para os estados do cuRAND (um estado por thread)
    err = cudaMalloc(&d_states, N*M*sizeof(curandState));
    if(err != cudaSuccess){
        printf("Erro de alocacao de memoria para curandState: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copiando matrizes do host para o device
    // d_matI não precisa ser copiado, será preenchido na primeira iteração
    err = cudaMemcpy(d_matP, h_matP, N*M*sizeof(int), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        printf("Erro na copia de d_matP: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemset(d_deaths, 0, sizeof(int));
    if(err != cudaSuccess){
        printf("Erro no cudaMemset de deaths: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemset(d_survivors, 0, sizeof(int));
    if(err != cudaSuccess){ 
        printf("Erro no cudaMemset de survivors: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Inicializa os geradores de números aleatórios
    unsigned long seed = (unsigned long)time(NULL);
    printf("Inicializando geradores aleatorios...\n");
    setup_curand<<<1, N*M>>>(d_states, seed, N, M);
    CUDA_CHECK(cudaGetLastError());
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("Erro na inicializacao do cuRAND: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Executa o kernel principal
    printf("Executando simulacao...\n");
    kernel<<<1, N*M>>>(d_matP, d_matI, d_deaths, d_survivors, N, M, N*M, d_states);
    CUDA_CHECK(cudaGetLastError());

    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("Erro na execucao do kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Verifica onde o resultado final ficou baseado no número de iterações
    if((N*M) % 2 == 0){
        printf("ENTROU NA PAR\n");
        // Número par de iterações: resultado em matP
        err = cudaMemcpy(h_matP, d_matP, N*M*sizeof(int), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("Erro na copia de d_matP para h_matP: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    else{
        printf("ENTROU NA IMPAR\n");
        // Número ímpar de iterações: resultado em matI
        err = cudaMemcpy(h_matP, d_matI, N*M*sizeof(int), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("Erro na copia de d_matI para h_matP: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }

    err = cudaMemcpy(h_survivors, d_survivors, sizeof(int), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        printf("Erro na copia de sobreviventes: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(h_deaths, d_deaths, sizeof(int), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        printf("Erro na copia de mortes: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Printando a matriz final (DEBUG)
    for(int i=0;i<N*M;i++){
        printf("\t[%d] ", h_matP[i]);
        if((i+1)%M==0)
            printf("\n");
    }
    
    FILE *fileOutput = fopen("infected_gpu.txt", "w");

    if(fileOutput == NULL){
        printf("Erro ao abrir o arquivo de saida.\n");
        return 1;
    }

    fprintf(fileOutput, "Mortos: %d, Sobreviventes: %d\n", *h_deaths, *h_survivors);
    fclose(fileOutput);

    // Libera memória do device
    cudaFree(d_matI);
    cudaFree(d_matP);
    cudaFree(d_states);
    cudaFree(d_deaths);
    cudaFree(d_survivors);

    return 0;
}