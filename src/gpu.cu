#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

// Kernel para inicializar os estados do cuRAND
__global__ void setup_curand(curandState *state, unsigned long seed, int N, int M) {
    int tid = threadIdx.x;
    if (tid < N * M) {
        // Inicializa o gerador para cada thread com seed única
        curand_init(seed, tid, 0, &state[tid]);
    }
}

__device__ void heal(int *matIn, int *matOut,int x, curandState *state){
    if (matIn[x] != -1){
        matOut[x] = matIn[x];
        return;
    }
    
    // Gera número aleatório de 0 a 9999 usando cuRAND
    unsigned int val = curand(state) % 10000;
    
    if (val < 1000)
        matOut[x] = 1;      // Fica saudável
    else if (val < 4000)    
        matOut[x] = -1;     // Continua infectado
    else                    
        matOut[x] = -2;     // Morre
}

__device__ void contaminate(int *matIn, int *matOut, int x, int N, int M){
    if (matIn[x]!=1) {
        matOut[x] = matIn[x];  // Se não é saudável, mantém o valor
        return;
    }

    if
    (
        (x%M > 0       && matIn[x-1] < 0) ||
        (x%M < M-1     && matIn[x+1] < 0) ||
        (x/M < N-1     && matIn[x+M] < 0) ||
        (x/M > 0       && matIn[x-M] < 0)
    ){
        matOut[x] = -1;  // Contamina
    }
    else {
        matOut[x] = matIn[x];  // Mantém saudável se não há vizinho infectado
    }
}

__device__ void removeDead(int *matIn, int *matOut, int x){
    if(matIn[x]==-2)
        matOut[x]=-3;
    else if(matIn[x]==-3)
        matOut[x]=0;
    else
        matOut[x]=matIn[x];  // Mantém o valor
}

__global__ void kernel(int *matP, int *matI, int N, int M, int max_iter, curandState *states){
    
    // Criando Threads
    int tid = threadIdx.x;
    if(tid >= N*M) return;

    // Estado local do gerador aleatório para esta thread
    curandState localState = states[tid];

    int parity=0;

    for(int i=0;i<max_iter;i++){
        if(parity==0){
            // Lê de matP, processa, escreve em matI
            contaminate(matP, matI, tid, N, M);
            __syncthreads();

            heal(matP, matI, tid, &localState);
            __syncthreads();

            removeDead(matP, matI, tid);
            __syncthreads();

            parity=1;
        }
        else{
            // Lê de matI, processa, escreve em matP
            contaminate(matI, matP, tid, N, M);
            __syncthreads();

            heal(matI, matP, tid, &localState);
            __syncthreads();

            removeDead(matI, matP, tid);
            __syncthreads();

            parity=0;
        }

        __syncthreads();
    }
    
    // Salva o estado atualizado de volta
    states[tid] = localState;
}

int main(void){
    
    // Declarando variaveis de dimensoes
    int N, M;

    // Abrindo arquivo da matriz de entrada
    FILE *fileInput = fopen("matriz_inicial.txt", "r");
    if(fileInput == NULL){
        printf("Erro ao abrir o arquivo de entrada.\n");
        return 1;
    }

    // Atribuindo as dimensões da matriz
    fscanf(fileInput, "%d", &N);
    fscanf(fileInput, "%d", &M);

    // Declarando matrizes no host
    int h_matI[N*M] = {0};
    int h_matP[N*M] = {0};

    // Atribuindo valores as matrizes do host
    for(int i=0;i<N*M;i++){
        fscanf(fileInput, "%d", &h_matP[i]);
    }
    fclose(fileInput);

    // Printando a matriz inicial (DEBUG)
    for(int i=0;i<N*M;i++){
        printf("\t[%d] ", h_matP[i]);
        if((i+1)%M==0)
            printf("\n");
    }
    // ----------------------------------------------------

    // Declarando matrizes no device
    int *d_matI, *d_matP;
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
    
    // Inicializa os geradores de números aleatórios
    unsigned long seed = (unsigned long)time(NULL);
    printf("Inicializando geradores aleatorios...\n");
    setup_curand<<<1, N*M>>>(d_states, seed, N, M);
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("Erro na inicializacao do cuRAND: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Executa o kernel principal
    printf("Executando simulacao...\n");
    kernel<<<1, N*M>>>(d_matP, d_matI, N, M, N*M, d_states);

    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("Erro na execucao do kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Verifica onde o resultado final ficou baseado no número de iterações
    int max_iter = N*M;
    if(max_iter % 2 == 0){
        // Número par de iterações: resultado em matP
        err = cudaMemcpy(h_matP, d_matP, N*M*sizeof(int), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("Erro na copia de d_matP para h_matP: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    else{
        // Número ímpar de iterações: resultado em matI
        err = cudaMemcpy(h_matP, d_matI, N*M*sizeof(int), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("Erro na copia de d_matI para h_matP: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }

    // Printando a matriz final (DEBUG)
    for(int i=0;i<N*M;i++){
        printf("\t[%d] ", h_matP[i]);
        if((i+1)%M==0)
            printf("\n");
    }
    // -----------------------------------------------------

    // Libera memória do device
    cudaFree(d_matI);
    cudaFree(d_matP);
    cudaFree(d_states);

    return 0;
}