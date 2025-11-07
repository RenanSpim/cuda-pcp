# Projeto 3: Simulação de Epidemia em CUDA
## Programação Concorrente e Paralela

---

### 1. Definição do Problema

Simulação de uma doença contagiosa em uma região retangular NxM.
Estados da matriz:
* `1`: Pessoa Saudável
* `-1`: Pessoa Contaminada
* `-2`: Pessoa Morta
* `0`: Ninguém

### 2. Regras da Simulação (Por Iteração)

1.  **Contaminação:**
    * `Saudável (1)` é contaminado se tiver vizinho (horizontal/vertical) `Contaminado (-1)` ou `Morto (-2)`.

2.  **Evolução (Contaminado):**
    * Uma `Pessoa Contaminada (-1)` tem seu destino decidido aleatoriamente (`rand() % 10000`):
        * `0 - 999` (10%): Cura -> `Saudável (1)`
        * `1000 - 3999` (30%): Permanece `Contaminado (-1)`
        * `> 4000` (60%): Morre -> `Morto (-2)`

3.  **Remoção (Morto):**
    * `Pessoa Morta (-2)` contamina por 1 iteração e depois vira `Ninguém (0)`.

4.  **Fim:**
    * Após `N * M` iterações ou quando não houver mais pessoas (saudáveis ou contaminadas).

### 3. O que deve ser feito

Implementar uma versão para **CPU** e uma para **GPU (CUDA)**. A simulação aplica as regras em uma matriz `i` para gerar a matriz `i+1`.

#### 3.1 Entrada

Arquivo único:
* Linha 1: `N` e `M` (inteiros)
* N Linhas seguintes: M inteiros (estado inicial)

#### 3.2 Saída

Arquivo único:
* Total de mortos e total de sobreviventes (contaminados + saudáveis).

### 4. Condições de Teste (Medir tempo médio de 3 execuções)

1.  Apenas CPU (sem GPU).
2.  GPU: 1 kernel em 1 bloco.
3.  GPU: N kernels em 1 bloco.
4.  GPU: N kernels em 2 blocos.
5.  GPU: N kernels em 4 blocos.
6.  GPU: N kernels em 8 blocos.
7.  GPU: N kernels em N blocos (1 kernel/bloco).
8.  GPU: N kernels em M blocos (N/M kernels/bloco).

### 5. Entregáveis

* Códigos-fonte (CPU e GPU).
* Relatório comparativo dos testes.

**Data de Entrega:** 24/11/2025