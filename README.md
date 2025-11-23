# Projeto 3: Simulação de Epidemia em CUDA
## Programação Concorrente e Paralela

---

## 1. Estrutura do Projeto

```
cuda-pcp/
├── bin/                      # Executáveis compilados
│   ├── cpu                   # Executável da versão CPU
│   └── gpu                   # Executável da versão GPU
├── data/                     # Dados de entrada
│   └── matriz_inicial.txt    # Matriz inicial da simulação
├── results/                  # Resultados das execuções
│   ├── infected_cpu.txt      # Resultado da versão CPU
│   ├── infected_gpu.txt      # Resultado da versão GPU
│   └── resultados_medias.txt # Comparativo de desempenho
├── src/                      # Códigos-fonte
│   ├── cpu.cu                # Implementação CPU
│   └── gpu.cu                # Implementação GPU (CUDA)
├── test_all.sh               # Script de teste automatizado
└── README.md                 # Este arquivo
```

---

## 2. Definição do Problema

Simulação de uma doença contagiosa em uma região retangular NxM.

### Estados da Matriz
* `1`: Pessoa Saudável
* `-1`: Pessoa Contaminada
* `-2`: Pessoa Morta
* `0`: Ninguém

---

## 3. Regras da Simulação

A cada iteração, as seguintes regras são aplicadas:

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

---

## 4. Como Usar

### Compilação Manual

```bash
# Compilar versão CPU
nvcc -o bin/cpu src/cpu.cu

# Compilar versão GPU
nvcc -o bin/gpu src/gpu.cu -arch=sm_75
```

### Execução Manual

```bash
# Executar versão CPU
./bin/cpu data/matriz_inicial.txt

# Executar versão GPU
./bin/gpu
# (Selecione o caso de teste no menu interativo)
```

### Teste Automatizado

O script `test_all.sh` compila e testa todos os casos automaticamente, calculando médias de 3 execuções:

```bash
./test_all.sh
```

**O que o script faz:**
- Compila as versões CPU e GPU
- Executa a versão CPU 3 vezes e calcula a média
- Executa todos os casos GPU (2-8) 3 vezes cada
- Calcula speedup e diferença percentual em relação à CPU
- Gera relatório completo em `results/resultados_medias.txt`

---

## 📊 Casos de Teste

O sistema suporta os seguintes casos de teste:

| Caso | Descrição |
|------|-----------|
| 1 | CPU (baseline) |
| 2 | GPU: 1 kernel em 1 bloco |
| 3 | GPU: n kernels em 1 bloco |
| 4 | GPU: n kernels em 2 blocos |
| 5 | GPU: n kernels em 4 blocos |
| 6 | GPU: n kernels em 8 blocos |
| 7 | GPU: n kernels em n blocos (1 kernel/bloco) |
| 8 | GPU: n kernels em m blocos (n/m kernels/bloco) |

---

## 5. Formato de Entrada

Arquivo de entrada (`data/matriz_inicial.txt`):
* **Linha 1:** `N` e `M` (dimensões da matriz - inteiros)
* **N linhas seguintes:** M inteiros separados por espaço (estado inicial de cada célula)

**Exemplo:**
```
10 20
1 1 0 0 -1 0 0 ...
0 1 1 0 0 0 1 ...
...
```

---

## 6. Formato de Saída

Os arquivos de saída contêm:
* Configuração da simulação (dimensões, tempo de execução)
* Estatísticas da população:
  - População inicial
  - Sobreviventes (saudáveis + infectados)
  - Mortos
  - Taxa de mortalidade
  - Taxa de sobrevivência

**Arquivos gerados:**
* `results/infected_cpu.txt` - Resultado da versão CPU
* `results/infected_gpu.txt` - Resultado da versão GPU
* `results/resultados_medias.txt` - Comparativo de desempenho

---

## 7. Métricas de Desempenho

O script de teste calcula:
* **Tempo médio** de execução (3 execuções)
* **Speedup**: razão entre tempo CPU e tempo GPU
* **Diferença percentual**: quanto mais rápido/lento em relação à CPU

---

## 8. Requisitos

* CUDA Toolkit
* GPU compatível com CUDA
* Compilador nvcc
* Sistema Linux/Unix com bash

---