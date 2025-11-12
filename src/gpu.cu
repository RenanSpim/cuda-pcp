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

__device__ void heal(int **mat, int x, int y, curandState *state){
    if (mat[x][y] != -1) return;
    
    // Gera número aleatório de 0 a 9999 usando cuRAND
    unsigned int val = curand(state) % 10000;
    
    if (val < 1000)         
        mat[x][y] = 1;      // Fica saudável
    else if (val < 4000)    
        mat[x][y] = -1;     // Continua infectado
    else                    
        mat[x][y] = -2;     // Morre
}

__device__ void contaminate(int **mat, int x, int y, int N, int M){
    if (mat[x][y]!=1) return;

    if
    (
        x > 0       && mat[x-1][y] < 0 ||
        x < (N-1)   && mat[x+1][y] < 0 ||
        y > 0       && mat[x][y-1] < 0 ||
        y < (M-1)   && mat[x][y+1] < 0
    ){
        mat[x][y] = -1;
    }
}

__device__ void removeDead(int **mat, int x, int y){
    if(mat[x][y]==-2)
        mat[x][y]=-3;
    else if(mat[x][y]==-3)
        mat[x][y]=0;
}

__global__ void kernel(int **matP, int **matI, int N, int M, int max_iter, curandState *states){
    
    // Criando Threads
    int tid = threadIdx.x;
    if(tid >= N*M) return;

    int x = (int)(tid / N);
    int y = (int)(tid % M);

    // Estado local do gerador aleatório para esta thread
    curandState localState = states[tid];

    int parity=0;

    for(int i=0;i<max_iter;i++){
        if(parity==0){
            contaminate(matP, x, y, N, M);
            __syncthreads();

            heal(matP, x, y, &localState);
            __syncthreads();

            removeDead(matP, x, y);
            __syncthreads();

            matI[x][y] = matP[x][y];
            parity=1;
        }
        else{
            contaminate(matI, x, y, N, M);
            __syncthreads();

            heal(matI, x, y, &localState);
            __syncthreads();

            removeDead(matI, x, y);
            __syncthreads();

            matP[x][y] = matI[x][y];
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
    int h_matI[N][M] = {0};
    int h_matP[N][M] = {0};

    // Atribuindo valores as matrizes do host
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            fscanf(fileInput, "%d", &h_matP[i][j]);
        }
    }
    fclose(fileInput);

    // Printando a matriz inicial (DEBUG)
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("\t[%d] ", h_matP[i][j]);
        }
        printf("\n");
    }
    // ----------------------------------------------------

    // Declarando matrizes no device
    int **d_matI, **d_matP;
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
    err = cudaMemcpy(d_matI, h_matI, N*M*sizeof(int), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        printf("Erro na copia de d_matI: %s\n", cudaGetErrorString(err));
        return 1;
    }
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

    // Retornando os resultados ao host
    err = cudaMemcpy(h_matP, d_matP, N*M*sizeof(int), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        printf("Erro na copia de d_matP para h_matP: %s\n", cudaGetErrorString(err));
        return 1;
    }   
    
    printf("Simulacao concluida!\n");

    // Printando a matriz final (DEBUG)
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("\t[%d] ", h_matP[i][j]);
        }
        printf("\n");
    }
    // -----------------------------------------------------

    // Libera memória do device
    cudaFree(d_matI);
    cudaFree(d_matP);
    cudaFree(d_states);

    return 0;
}