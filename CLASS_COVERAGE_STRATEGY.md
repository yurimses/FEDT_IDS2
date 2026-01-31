# Estratégia de Cobertura por Classe (Class Coverage)

## Descrição

A estratégia `class_coverage` foi implementada para melhorar o desempenho do FEDT em cenários Non-IID onde alguns clientes podem ter classes ausentes. Esta abordagem garante que o ensemble global contenha "especialistas" para cada classe.

## Como Funciona

1. **Avaliação por Classe**: Para cada árvore recebida dos clientes, calcula-se o F1-score para cada classe individualmente (usando `f1_score(..., average=None, labels=ALL_LABELS)`).

2. **Seleção de Especialistas**: Para cada classe, seleciona as top-N árvores com melhor F1-score para aquela classe específica.

3. **Complementação**: Após garantir especialistas para cada classe, completa o ensemble com as melhores árvores por macro-F1 até atingir o número total desejado.

## Configuração

No arquivo `fedt/config.toml`:

```toml
[settings]
aggregation_strategy = "class_coverage"

[settings.server]
# Número de árvores especialistas a selecionar por classe
trees_per_class = 3

# Proporção total de árvores a manter (0.5 = 50%)
total_trees_ratio = 0.5
```

## Parâmetros

- **`trees_per_class`** (default: 3): Número de árvores especialistas que serão selecionadas para cada classe. Valores típicos: 2-5.

- **`total_trees_ratio`** (default: 0.5): Proporção do total de árvores recebidas que será mantida no ensemble final. Por exemplo, se recebermos 200 árvores e `total_trees_ratio=0.5`, o ensemble terá até 100 árvores.

## Quando Usar

Esta estratégia é **especialmente recomendada** para:

- ✅ Cenários **Non-IID** onde diferentes clientes têm distribuições de classes muito diferentes
- ✅ Datasets com **classes raras** ou desbalanceadas
- ✅ Situações onde alguns clientes têm **classes ausentes**
- ✅ Quando se busca melhorar o desempenho em **todas as classes**, não apenas nas majoritárias

## Comparação com Outras Estratégias

| Estratégia | Vantagens | Desvantagens |
|------------|-----------|--------------|
| `random` | Simples, rápido | Não considera qualidade |
| `best_trees` | Seleciona melhores por macro-F1 | Pode ignorar classes raras |
| `threshold` | Filtra árvores ruins (MCC) | Pode perder especialistas de classes raras |
| `best_forests` | Mantém coesão da floresta | Descarta florestas potencialmente boas |
| **`class_coverage`** | **Garante cobertura de todas as classes** | **Mais computacionalmente custoso** |

## Exemplo de Uso

### 1. Configurar no `config.toml`

```toml
[settings]
number_of_clients = 10
number_of_rounds = 20
aggregation_strategy = "class_coverage"

[settings.server]
trees_per_class = 3
total_trees_ratio = 0.5

[dataset]
partition_type = "non_iid"
non_iid_alpha = 0.3
```

### 2. Executar o servidor

```bash
source env-fedt-ids/bin/activate
fedt server
```

### 3. Executar os clientes (em outro terminal)

```bash
source env-fedt-ids/bin/activate
fedt clients
```

## Saída Esperada

Durante a agregação, você verá logs como:

```
[COBERTURA POR CLASSE] Avaliando 200 árvores...
[COBERTURA POR CLASSE] Selecionando 3 árvores para cada uma das 15 classes...
  Classe DDoS_HTTP: melhor F1=0.9845
  Classe DDoS_TCP: melhor F1=0.9712
  Classe DDoS_UDP: melhor F1=0.9823
  ...
[COBERTURA POR CLASSE] 45 árvores únicas selecionadas como especialistas
[COBERTURA POR CLASSE] Completando até 100 árvores com melhores macro-F1...
[COBERTURA POR CLASSE] Total final: 100 árvores selecionadas
[COBERTURA POR CLASSE] Macro-F1 médio das árvores selecionadas: 0.8234
```

## Ajuste Fino

### Para datasets pequenos
```toml
trees_per_class = 2
total_trees_ratio = 0.6
```

### Para muitas classes (>20)
```toml
trees_per_class = 2
total_trees_ratio = 0.4
```

### Para melhor qualidade (mais árvores especialistas)
```toml
trees_per_class = 5
total_trees_ratio = 0.7
```

## Comparação Experimental

Para comparar todas as estratégias, use:

```toml
[settings.sequence]
many_simulations = true
number_of_simulations = 1
aggregation_strategies = ['random', 'best_trees', 'threshold', 'class_coverage']
```

Os resultados ficarão em `results/` com análise comparativa.

## Referências

Esta implementação é baseada em técnicas comuns de ensemble learning:
- Diversity in ensemble classifiers
- Specialist models for imbalanced datasets
- Class-specific feature selection

## Notas de Implementação

- A estratégia usa `np.unique()` para garantir que a mesma árvore não seja selecionada múltiplas vezes
- O conjunto de validação do servidor é usado para avaliar as árvores
- O tempo de agregação aumenta proporcionalmente ao número de árvores e classes
