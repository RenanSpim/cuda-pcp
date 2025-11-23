#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Função para imprimir uma linha da tabela
void print_table_line(const char* label, const char* value) {
    printf("| %-25s | %-30s |\n", label, value);
}

void print_table_separator() {
    printf("+---------------------------+--------------------------------+\n");
}

// Kernel para inicializar os estados do cuRAND
__global__ void setup_curand(curandState *state, unsigned long seed, int N, int M) {
    int tid = threadIdx.x;
    if (tid < N * M) {
        // Inicializa o gerador para cada thread com seed única
        curand_init(seed, tid, 0, &state[tid]);
    }
}

__global__ void kernel_1thread(int *matP, int *matI, int *deaths, int *survivors, int N, int M, int max_iter, curandState *states, int *has_living, int *has_infected){
    
    // Estado local do gerador aleatório
    curandState localState = states[0];

    // Ponteiros para matriz de entrada e saída
    int *matIn = matP;
    int *matOut = matI;

    for(int i=0; i<max_iter; i++){
        // Reseta os flags no início de cada iteração
        *has_living = 0;
        *has_infected = 0;
        
        // Determina a matriz de entrada e saída para esta iteração
        int parity = i % 2;
        if (parity == 0) {
            matIn = matP;
            matOut = matI;
        } else {
            matIn = matI;
            matOut = matP;
        }

        // Processa todas as células
        for(int tid=0; tid<N*M; tid++){
            
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
                else{
                    out_val = -2;    // Morre
                    (*deaths)++;
                }
            } // RemoveDead
            else if (in_val == -2) { // Morto (primeira iteracao
                out_val = -3;
            } 
            else if (in_val == -3) { // Morto (segunda iteração)
                out_val = 0;
            }
            
            // Escreve na matOut
            matOut[tid] = out_val;
            
            // Verifica se ainda há população viva (saudável ou infectada)
            if (out_val == 1 || out_val == -1) {
                (*has_living)++;
            }
            
            // Verifica se ainda há infectados
            if (out_val == -1) {
                (*has_infected)++;
            }
        }
        
        // Para a simulação se não há mais população viva OU se não há mais infectados (todos curados)
        if(*has_living == 0 || *has_infected == 0){
            break;
        }
    }

    // Contagem de sobreviventes (infectados e saudáveis)
    for(int tid=0; tid<N*M; tid++){
        if (matOut[tid] != 0 && matOut[tid] > -2) {
            (*survivors)++;
        }
    }
    
    // Salva o estado atualizado de volta
    states[0] = localState;
}

__global__ void kernel(int *matP, int *matI, int *deaths, int *survivors, int N, int M, int max_iter, curandState *states, int *has_living, int *has_infected){
    
    //sincronizando os blocos
    cg::grid_group grid = cg::this_grid();

    // Criando Threads
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= N*M) return;

    // Estado local do gerador aleatório
    curandState localState = states[tid];

    // Ponteiros para matriz de entrada e saída
    int *matIn = matP;
    int *matOut = matI;

    for(int i=0; i<max_iter; i++){
        
        // Reseta os flags no início de cada iteração
        if(tid == 0){
            *has_living = 0;
            *has_infected = 0;
        }
        
        // Determina a matriz de entrada e saída para esta iteração
        int partiy  = i % 2;
        if (partiy == 0) {
            matIn = matP;
            matOut = matI;
        } else {
            matIn = matI;
            matOut = matP;
        }

        // Sincroniza todos os blocos
        grid.sync();

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
            else{
                out_val = -2;    // Morre
                atomicAdd(deaths, 1);
            }
        } // RemoveDead
        else if (in_val == -2) { // Morto (primeira iteracao
            out_val = -3;
        } 
        else if (in_val == -3) { // Morto (segunda iteração)
            out_val = 0;
        }
        
        // Cada thread escreve na matOut
        matOut[tid] = out_val;
        
        // Verifica se ainda há população viva (saudável ou infectada)
        if (out_val == 1 || out_val == -1) {
            atomicAdd(has_living, 1);
        }
        
        // Verifica se ainda há infectados
        if (out_val == -1) {
            atomicAdd(has_infected, 1);
        }
        
        // Sincroniza todos os blocos após escrever
        grid.sync();
        
        // Para a simulação se não há mais população viva OU se não há mais infectados (todos curados)
        if(*has_living == 0 || *has_infected == 0){
            break;
        }
    }

    // Contagem de sobrevivente (infectados e saudáveis)
    if (matOut[tid] != 0 && matOut[tid] > -2) {
        atomicAdd(survivors, 1);
    }
    
    // Salva o estado atualizado de volta
    states[tid] = localState;
}

int main(void){
    
    // Declarando variaveis de dimensoes
    int N, M;
    int *h_survivors = (int*)malloc(sizeof(int));
    int *h_deaths = (int *)malloc(sizeof(int));
    *h_survivors = 7;
    *h_deaths = 7;

    // Abrindo arquivo da matriz de entrada
    FILE *fileInput = fopen("data/matriz_inicial.txt", "r");
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
    int *d_matI, *d_matP, *d_deaths, *d_survivors, *d_has_living, *d_has_infected;
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
        printf("Erro de alocacao de memoria para d_deaths: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_survivors, sizeof(int));
    if(err != cudaSuccess){
        printf("Erro de alocacao de memoria para d_survivors: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_has_living, sizeof(int));
    if(err != cudaSuccess){
        printf("Erro de alocacao de memoria para d_has_living: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_has_infected, sizeof(int));
    if(err != cudaSuccess){
        printf("Erro de alocacao de memoria para d_has_infected: %s\n", cudaGetErrorString(err));
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
    err = cudaMemset(d_has_living, 0, sizeof(int));
    if(err != cudaSuccess){ 
        printf("Erro no cudaMemset de has_living: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemset(d_has_infected, 0, sizeof(int));
    if(err != cudaSuccess){ 
        printf("Erro no cudaMemset de has_infected: %s\n", cudaGetErrorString(err));
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

    // Seção de coleta de dados para o teste
    int resposta;
    printf("Qual o modelo de teste que deseja executar?\n");
    printf("2 - 1 Kernel em 1 bloco\n");
    printf("3 - n kernels em 1 bloco\n");
    printf("4 - n kernels em 2 bloco\n");
    printf("5 - n kernels em 4 bloco\n");
    printf("6 - n kernels em 8 bloco\n");
    printf("7 - n kernels em n blocos (1 kernel por bloco)\n");
    printf("8 - n kernels em n blocos (n/m kernel por bloco)\n");
    printf("0 - SAIR\n");
    scanf("%d", &resposta);

    // Declarando variáveis para medição de tempo
    struct timespec start, end;
    double tempo_total;

    // Configuração para lançamento cooperativo
    int numBlocks, threadsPerBlock;
    
    switch(resposta){
        case 0:
            printf("Saindo...\n");
            return 0;
        case 2:
            numBlocks = 1;
            threadsPerBlock = 1;
            break;
        case 3:
            numBlocks = 1;
            threadsPerBlock = N*M;
            break;
        case 4:
            numBlocks = 2;
            threadsPerBlock = (N*M)/2;
            break;
        case 5:
            numBlocks = 4;
            threadsPerBlock = (N*M)/4;
            break;
        case 6:
            numBlocks = 8;
            threadsPerBlock = (N*M)/8;
            break;
        case 7:
            numBlocks = N*M;
            threadsPerBlock = 1;
            break;
        case 8:
            numBlocks = N;
            threadsPerBlock = M;
            break;
        default:
            printf("Opcao invalida. Saindo...\n");
            return 1;
    }

    // Preparar argumentos do kernel
    int max_iterations = N*M;
    void *kernelArgs[] = {
        (void*)&d_matP,
        (void*)&d_matI,
        (void*)&d_deaths,
        (void*)&d_survivors,
        (void*)&N,
        (void*)&M,
        (void*)&max_iterations,
        (void*)&d_states,
        (void*)&d_has_living,
        (void*)&d_has_infected
    };

    // Executar o kernel cooperativo
    printf("Executando simulacao com %d blocos e %d threads por bloco...\n", numBlocks, threadsPerBlock);
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    if(resposta == 2){
        err = cudaLaunchCooperativeKernel(
            (void*)kernel_1thread,
            dim3(numBlocks),
            dim3(threadsPerBlock),
            kernelArgs,
            0,
            NULL
        );
    }
    else{
        err = cudaLaunchCooperativeKernel(
        (void*)kernel,
        dim3(numBlocks),
        dim3(threadsPerBlock),
        kernelArgs,
        0,
        NULL
        );
    }
    
    if(err != cudaSuccess){
        printf("Erro no lancamento cooperativo do kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("Erro na execucao do kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    tempo_total = (end.tv_sec - start.tv_sec);
    tempo_total += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    printf("\n=== Simulacao concluida com sucesso! ===\n");
    printf("Tempo de execucao: %.6f segundos\n\n", tempo_total);

    // Verifica onde o resultado final ficou baseado no número de iterações
    if((N*M) % 2 == 0){
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

    // Contagem de população final por estado
    int saudaveis = 0, infectados = 0, mortos_recentes = 0, espacos_vazios = 0;
    for(int i=0; i<N*M; i++){
        if(h_matP[i] == 1) saudaveis++;
        else if(h_matP[i] == -1) infectados++;
        else if(h_matP[i] == -2 || h_matP[i] == -3) mortos_recentes++;
        else if(h_matP[i] == 0) espacos_vazios++;
    }
    
    int populacao_inicial = saudaveis + infectados + *h_deaths;
    double taxa_mortalidade = (populacao_inicial > 0) ? (100.0 * (*h_deaths) / populacao_inicial) : 0.0;
    double taxa_sobrevivencia = (populacao_inicial > 0) ? (100.0 * (*h_survivors) / populacao_inicial) : 0.0;
    
    // Exibir tabela de resultados
    printf("\n");
    print_table_separator();
    printf("|     RESULTADOS DA SIMULACAO - GPU                          |\n");
    print_table_separator();
    
    char buffer[50];
    sprintf(buffer, "%d x %d", N, M);
    print_table_line("Dimensoes da Matriz", buffer);
    
    sprintf(buffer, "%d", N*M);
    print_table_line("Tamanho Total", buffer);
    
    sprintf(buffer, "%d blocos x %d threads", numBlocks, threadsPerBlock);
    print_table_line("Configuracao GPU", buffer);
    
    sprintf(buffer, "%.6f segundos", tempo_total);
    print_table_line("Tempo de Execucao", buffer);
    
    
    print_table_separator();
    printf("|     ESTATISTICAS DA POPULACAO                              |\n");
    print_table_separator();
    
    sprintf(buffer, "%d", populacao_inicial);
    print_table_line("Populacao Inicial", buffer);
    
    sprintf(buffer, "%d", *h_survivors);
    print_table_line("Sobreviventes", buffer);
    
    sprintf(buffer, "%d", saudaveis);
    print_table_line("  - Saudaveis", buffer);
    
    sprintf(buffer, "%d", infectados);
    print_table_line("  - Infectados", buffer);
    
    sprintf(buffer, "%d", *h_deaths);
    print_table_line("Mortes", buffer);
    
    sprintf(buffer, "%.2f%%", taxa_mortalidade);
    print_table_line("Taxa de Mortalidade", buffer);
    
    sprintf(buffer, "%.2f%%", taxa_sobrevivencia);
    print_table_line("Taxa de Sobrevivencia", buffer);
    
    print_table_separator();
    
    // Printando a matriz final (DEBUG)
    printf("\nMatriz Final:\n");
    for(int i=0;i<N*M;i++){
        printf("\t[%d] ", h_matP[i]);
        if((i+1)%M==0)
            printf("\n");
    }
    printf("\n");
    
    // Salvar resultados em arquivo
    FILE *fileOutput = fopen("infected_gpu.txt", "w");

    if(fileOutput == NULL){
        printf("Erro ao abrir o arquivo de saida.\n");
        return 1;
    }

    fprintf(fileOutput, "=== RESULTADOS DA SIMULACAO - GPU ===\n\n");
    fprintf(fileOutput, "Configuracao:\n");
    fprintf(fileOutput, "  Dimensoes: %d x %d\n", N, M);
    fprintf(fileOutput, "  Blocos: %d\n", numBlocks);
    fprintf(fileOutput, "  Threads por bloco: %d\n", threadsPerBlock);
    fprintf(fileOutput, "  Tempo de execucao: %.6f segundos\n\n", tempo_total);
    fprintf(fileOutput, "Estatisticas:\n");
    fprintf(fileOutput, "  Populacao Inicial: %d\n", populacao_inicial);
    fprintf(fileOutput, "  Sobreviventes: %d\n", *h_survivors);
    fprintf(fileOutput, "    - Saudaveis: %d\n", saudaveis);
    fprintf(fileOutput, "    - Infectados: %d\n", infectados);
    fprintf(fileOutput, "  Mortos: %d\n", *h_deaths);
    fprintf(fileOutput, "  Taxa de Mortalidade: %.2f%%\n", taxa_mortalidade);
    fprintf(fileOutput, "  Taxa de Sobrevivencia: %.2f%%\n", taxa_sobrevivencia);
    fclose(fileOutput);

    printf("Resultados salvos em 'results/infected_gpu.txt'\n");    // Libera memória do device
    cudaFree(d_matI);
    cudaFree(d_matP);
    cudaFree(d_states);
    cudaFree(d_deaths);
    cudaFree(d_survivors);
    cudaFree(d_has_living);
    cudaFree(d_has_infected);

    return 0;
}