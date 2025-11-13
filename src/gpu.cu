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
    
}

__device__ void removeDead(int *matIn, int *matOut, int x, int *deaths){
    if(matIn[x]==-2){
        
    }
    else if(matIn[x]==-3)
        matOut[x]=0;
    else
        matOut[x]=matIn[x];  // Mantém o valor
}

__global__ void kernel(int *matP, int *matI, int *deaths, int *survivors, int N, int M, int max_iter, curandState *states){
    
    // Problema 2: Corrigido - Apenas a thread 0 inicializa
    if (threadIdx.x == 0) {
        *deaths = 0;
    }
    __syncthreads();

    // Criando Threads
    int tid = threadIdx.x;
    if(tid >= N*M) return;

    // Estado local do gerador aleatório
    curandState localState = states[tid];
    int parity = 0;

    // Ponteiros para matriz de entrada e saída
    int *matIn = matP;
    int *matOut = matI;

    for(int i=0; i<max_iter; i++){
        
        // Determina a matriz de entrada e saída para esta iteração
        if (parity == 0) {
            matIn = matP;
            matOut = matI;
        } else {
            matIn = matI;
            matOut = matP;
        }

        // Garante que todas as threads definiram seus ponteiros
        __syncthreads(); 

        // --- Problema 1: Corrigido - Lógica de estado mesclada ---
        
        int in_val = matIn[tid];
        int out_val = in_val; // Valor padrão é manter o estado

        // Lógica de Contaminate
        if (in_val == 1) { // Saudável
            if (
                (tid%M > 0    && matIn[tid-1] < 0) || // Vizinho esquerdo
                (tid%M < M-1  && matIn[tid+1] < 0) || // Vizinho direito
                (tid/M < N-1  && matIn[tid+M] < 0) || // Vizinho baixo
                (tid/M > 0    && matIn[tid-M] < 0)    // Vizinho cima
            ){
                out_val = -1; // Contamina
            }
            // else: continua 1 (já definido em out_val)
        } 
        // Lógica de Heal
        else if (in_val == -1) { // Infectado
            unsigned int val = curand(&localState) % 10000;
            if (val < 1000)
                out_val = 1;     // Fica saudável
            else if (val < 4000)
                out_val = -1;    // Continua infectado
            else
                out_val = -2;    // Morre
        } 
        // Lógica de RemoveDead
        else if (in_val == -2) { // Morto (frame 1)
            out_val = -3;
            atomicAdd(deaths, 1); // Conta a morte aqui
        } 
        else if (in_val == -3) { // Morto (frame 2)
            out_val = 0; // Desaparece
        }
        
        // Espera todas as threads terminarem de LER matIn
        __syncthreads();
        
        // Todas as threads escrevem o novo estado em matOut
        matOut[tid] = out_val;
        
        // Espera todas as threads terminarem de ESCREVER em matOut
        __syncthreads();
        
        // Inverte a paridade
        parity = 1 - parity;
    }
    
    // --- Problema 3: Corrigido - Contagem de sobreviventes DEPOIS do loop ---

    // Garante que a última iteração terminou para todas as threads
    __syncthreads();

    // Apenas a thread 0 zera o contador de sobreviventes
    if (tid == 0) {
        *survivors = 0;
    }
    
    // Garante que o contador foi zerado
    __syncthreads();

    // O ponteiro matIn agora aponta para a matriz final (devido à última troca de paridade)
    if (matIn[tid] > 0) { // Conta qualquer um que seja saudável (valor 1)
        atomicAdd(survivors, 1);
    }
    
    // Salva o estado atualizado de volta
    states[tid] = localState;
}

int main(void){
    
    // Declarando variaveis de dimensoes
    int N, M;
    int *h_survivors = (int*)malloc(sizeof(int));
    int *h_deaths = (int*)malloc(sizeof(int));

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
    int *d_matI, *d_matP, *deaths, *survivors;
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
    err = cudaMalloc(&deaths, sizeof(int));
    if(err != cudaSuccess){
        printf("Erro de alocacao de memoria para d_matP: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&survivors, sizeof(int));
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
    kernel<<<1, N*M>>>(d_matP, d_matI, deaths, survivors, N, M, N*M, d_states);

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

    err = cudaMemcpy(h_survivors, survivors, sizeof(int), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        printf("Erro na copia de sobreviventes: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(h_deaths, deaths, sizeof(int), cudaMemcpyDeviceToHost);
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

    printf("Numéro de mortes: %d\n", *h_deaths);
    printf("Numéro de sobreviventes: %d\n", *h_survivors);
    // -----------------------------------------------------

    // Libera memória do device
    cudaFree(d_matI);
    cudaFree(d_matP);
    cudaFree(d_states);
    cudaFree(deaths);
    cudaFree(survivors);

    return 0;
}