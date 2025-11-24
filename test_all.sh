#!/bin/bash

# Script para testar todos os casos e calcular média de tempo de 3 execuções

echo "========================================"
echo "  TESTE COMPLETO DE TODOS OS CASOS"
echo "========================================"
echo ""

# Compilar os programas
echo "Compilando programas GPU e CPU..."
nvcc -o bin/gpu src/gpu.cu -arch=sm_75
if [ $? -ne 0 ]; then
    echo "Erro na compilação da GPU!"
    exit 1
fi

nvcc -o bin/cpu src/cpu.cu -arch=sm_75
if [ $? -ne 0 ]; then
    echo "Erro na compilação da CPU!"
    exit 1
fi
echo "Compilação concluída com sucesso!"
echo ""

# Testar CPU primeiro
echo "========================================"
echo "  TESTANDO VERSÃO CPU"
echo "========================================"
echo ""

soma_cpu=0
tempos_cpu=()

for exec in 1; do
    echo "Execução CPU $exec/3..."
    output=$(./bin/cpu data/realdata.dat 2>&1)
    
    tempo=$(echo "$output" | grep -oP "Tempo de execucao: \K[0-9]+\.[0-9]+" | head -1)
    
    if [ -z "$tempo" ]; then
        echo "  Erro ao extrair tempo da execução CPU $exec"
        tempo=0
    else
        echo "  Tempo: $tempo segundos"
        tempos_cpu+=($tempo)
        soma_cpu=$(echo "$soma_cpu + $tempo" | bc -l)
    fi
done

media_cpu=$(echo "scale=6; $soma_cpu / ${#tempos_cpu[@]}" | bc -l)
media_cpu_ms=$(echo "scale=3; $media_cpu * 1000" | bc -l)

echo ""
echo "Tempo médio CPU: $media_cpu segundos ($media_cpu_ms ms)"
echo ""

# Array com os casos de teste (2 a 8)
casos=(2 3 4 5 6 7 8)
nomes=("1 Kernel em 1 bloco" "n kernels em 1 bloco" "n kernels em 2 blocos" "n kernels em 4 blocos" "n kernels em 8 blocos" "n kernels em n blocos (1 kernel por bloco)" "n kernels em n blocos (n/m kernel por bloco)")

# Arquivo de resultados
resultado_file="results/resultados_medias_big_inp.txt"
echo "=== RESULTADOS DAS MÉDIAS DE TEMPO ===" > $resultado_file
echo "Data: $(date)" >> $resultado_file
echo "" >> $resultado_file

echo "========================================" | tee -a $resultado_file
echo "TEMPO DE EXECUÇÃO - CPU" | tee -a $resultado_file
echo "========================================" | tee -a $resultado_file
echo "Execução 1: ${tempos_cpu[0]} s" | tee -a $resultado_file
echo "Execução 2: ${tempos_cpu[1]} s" | tee -a $resultado_file
echo "Execução 3: ${tempos_cpu[2]} s" | tee -a $resultado_file
echo "Média CPU: $media_cpu s ($media_cpu_ms ms)" | tee -a $resultado_file
echo "" | tee -a $resultado_file

echo "========================================" | tee -a $resultado_file
echo "COMPARAÇÃO GPU vs CPU" | tee -a $resultado_file
echo "========================================" | tee -a $resultado_file
echo "" >> $resultado_file

echo "+-----------------------------------------------------------------------------------------------------------+" | tee -a $resultado_file
echo "| Caso | Configuração                                  | Tempo Médio GPU | Speedup  | Diferença       |" | tee -a $resultado_file
echo "+-----------------------------------------------------------------------------------------------------------+" | tee -a $resultado_file

# Loop pelos casos de teste
for i in "${!casos[@]}"; do
    caso=${casos[$i]}
    nome="${nomes[$i]}"
    
    echo ""
    echo "----------------------------------------"
    echo "Testando Caso $caso: $nome"
    echo "----------------------------------------"
    
    soma=0
    tempos=()
    
    # Executar 3 vezes
    for exec in 1 2 3; do
        echo "  Execução $exec/3..."
        
        # Executar e capturar o tempo
        output=$(echo "$caso" | ./bin/gpu 2>&1)
        
        # Extrair o tempo em segundos (procura por "Tempo de execucao: X.XXXXXX segundos")
        tempo=$(echo "$output" | grep -oP "Tempo de execucao: \K[0-9]+\.[0-9]+" | head -1)
        
        if [ -z "$tempo" ]; then
            echo "    Erro ao extrair tempo da execução $exec"
            tempo=0
        else
            echo "    Tempo: $tempo segundos"
            tempos+=($tempo)
            soma=$(echo "$soma + $tempo" | bc -l)
        fi
    done
    
    # Calcular média
    if [ ${#tempos[@]} -gt 0 ]; then
        media=$(echo "scale=6; $soma / ${#tempos[@]}" | bc -l)
        media_ms=$(echo "scale=3; $media * 1000" | bc -l)
        
        # Calcular speedup (CPU / GPU)
        speedup=$(echo "scale=2; $media_cpu / $media" | bc -l)
        
        # Calcular diferença percentual
        if (( $(echo "$media_cpu > $media" | bc -l) )); then
            diff_percent=$(echo "scale=2; (($media_cpu - $media) / $media_cpu) * 100" | bc -l)
            diff_text=$(printf "%.2f%% mais rápido" "$diff_percent")
        else
            diff_percent=$(echo "scale=2; (($media - $media_cpu) / $media) * 100" | bc -l)
            diff_text=$(printf "%.2f%% mais lento" "$diff_percent")
        fi
        
        printf "| %-4s | %-45s | %10.6f s    | %7.2fx | %-15s |\n" "$caso" "$nome" "$media" "$speedup" "$diff_text" | tee -a $resultado_file
        
        echo ""
        echo "  Tempos individuais: ${tempos[@]}"
        echo "  Média: $media segundos ($media_ms ms)"
        echo "  Speedup: ${speedup}x em relação à CPU"
        echo "  $diff_text que a CPU"
        echo "" >> $resultado_file
        echo "Caso $caso - $nome" >> $resultado_file
        echo "  Execução 1: ${tempos[0]} s" >> $resultado_file
        echo "  Execução 2: ${tempos[1]} s" >> $resultado_file
        echo "  Execução 3: ${tempos[2]} s" >> $resultado_file
        echo "  Média: $media s ($media_ms ms)" >> $resultado_file
        echo "  Speedup: ${speedup}x" >> $resultado_file
        echo "  Diferença: $diff_text" >> $resultado_file
        echo "" >> $resultado_file
    else
        printf "| %-4s | %-45s | %15s | %8s | %15s |\n" "$caso" "$nome" "ERRO" "N/A" "N/A" | tee -a $resultado_file
        echo "  Erro: não foi possível calcular média"
    fi
done

echo "+-----------------------------------------------------------------------------------------------------------+" | tee -a $resultado_file
echo "" | tee -a $resultado_file
echo "========================================" | tee -a $resultado_file
echo "RESUMO" | tee -a $resultado_file
echo "========================================" | tee -a $resultado_file
echo "Tempo médio CPU: $media_cpu s" | tee -a $resultado_file
echo "" | tee -a $resultado_file
echo ""
echo "========================================"
echo "  TESTE COMPLETO FINALIZADO"
echo "========================================"
echo "Resultados salvos em: $resultado_file"
echo ""
