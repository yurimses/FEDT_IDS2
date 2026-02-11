# Machine Unlearning Implementation

## Overview
Este documento descreve a implementação de Machine Unlearning (esquecimento de dados) no sistema FEDT-IDS. O objetivo é permitir que as árvores de decisão de um cliente específico (cliente dominante) sejam removidas do modelo global após um determinado round, simulando a solicitação do cliente de ter seus dados esquecidos.

## Requisitos Implementados

### 1. Rastreio de Árvores por Cliente
**Status**: ✅ Implementado

Cada árvore é rastreada para saber qual cliente a enviou através de:
- `trees_warehouse`: lista de tuplas `(client_ID, client_trees)` 
- Armazena todas as árvores recebidas com sua origem (ID do cliente)
- Permite identificar e filtrar árvores específicas de um cliente

**Localização**: `server.py` - método `aggregate_trees()`

### 2. Exclusão de Árvores do Cliente Dominante
**Status**: ✅ Implementado

O servidor remove automaticamente todas as árvores do cliente dominante após o round de unlearning:

**Mudanças em `server.py`**:
- Método `aggregate_strategy()`: 
  - Verifica se `self.round >= self.unlearning_round`
  - Filtra `best_forests` removendo árvores onde `client_ID == self.dominant_client_id`
  - Marca `self.unlearning_done = True` para executar apenas uma vez
  - Registra em log: `[UNLEARNING] Árvores do cliente dominante removidas após round X`

**Localização**: `server.py` linhas 136-144

### 3. Bloqueio de Comunicação do Cliente Dominante
**Status**: ✅ Implementado

O cliente dominante deixa de se comunicar com o servidor após o round N (unlearning_round):

**Mudanças em `server.py`**:
- Método `aggregate_trees()`:
  - Verifica se `self.round >= self.unlearning_round` e se `client_ID == self.dominant_client_id`
  - Retorna imediatamente sem processar as árvores do cliente bloqueado
  - Registra em log: `[UNLEARNING] Ignorando árvores do cliente dominante após round X`

**Localização**: `server.py` linhas 203-206

### 4. Ajuste Dinâmico de Clientes Esperados
**Status**: ✅ Implementado

O servidor ajusta dinamicamente o número de clientes que espera em cada round:
- **Antes do unlearning_round**: Aguarda `number_of_clients` clientes
- **A partir do unlearning_round**: Aguarda `number_of_clients - 1` clientes

**Mudanças em `server.py`**:
1. Método `end_of_transmission()` (linhas 286-290):
   - Calcula `clientes_esperados_atual` dinamicamente
   - Se `self.round >= self.unlearning_round`: ajusta para `number_of_clients - 1`
   - Compara `clientes_respondidos == clientes_esperados_atual`

2. Método `_reset_server_sync()` (linhas 367-371):
   - Reseta `clientes_esperados` após cada round
   - Se `self.round >= self.unlearning_round`: ajusta para `number_of_clients - 1`

**Localização**: `server.py` linhas 286-290, 367-371

## Configuração

### Novos Parâmetros em `config.toml`

```toml
# [UNLEARNING] Parâmetros para machine unlearning
unlearning_round = 10  # Round após o qual o cliente dominante é esquecido e bloqueado
```

### Parâmetros Existentes Utilizados

```toml
# [CLASSIF] Parâmetros para estratégia 'dominant_client'
dominant_client_id = 0  # ID do cliente que terá a maior porcentagem dos dados
dominant_client_percentage = 0.7  # Porcentagem de dados do cliente dominante
```

## Fluxo de Execução

### Rounds 0 a (unlearning_round - 1)
1. Servidor aguarda `number_of_clients` clientes
2. Cliente dominante (ID=0) envia árvores normalmente
3. Todas as árvores são incluídas na agregação
4. Modelo global é atualizado com todas as árvores

### Round = unlearning_round
1. Servidor aguarda `number_of_clients - 1` clientes
2. Cliente dominante (ID=0) tenta enviar árvores mas é bloqueado
3. Servidor ignora qualquer comunicação do cliente dominante
4. Método `aggregate_strategy()` remove todas as árvores do cliente dominante de `trees_warehouse`
5. Modelo global é atualizado SEM as árvores do cliente dominante

### Rounds > unlearning_round
1. Servidor continua aguardando `number_of_clients - 1` clientes
2. Cliente dominante não participa mais do treinamento
3. Seu histórico é efetivamente removido do modelo

## Logs de Rastreamento

Durante a execução, os seguintes logs indicam o funcionamento correto:

```
[UNLEARNING] Ignorando árvores do cliente dominante (ID=0) após round 10.
[UNLEARNING] Árvores do cliente dominante (ID=0) removidas após round 10.
```

## Modifications Summary

| Arquivo | Linhas | Mudança |
|---------|--------|---------|
| `config.toml` | 75-76 | Adicionado `unlearning_round` |
| `settings.py` | 87 | Adicionado carregamento de `unlearning_round` |
| `server.py` | 17 | Importar `dominant_client_id` e `unlearning_round` |
| `server.py` | 68-70 | Inicializar variáveis de unlearning no `__init__` |
| `server.py` | 136-144 | Implementar exclusão de árvores em `aggregate_strategy()` |
| `server.py` | 203-206 | Implementar bloqueio em `aggregate_trees()` |
| `server.py` | 286-290 | Ajuste dinâmico em `end_of_transmission()` |
| `server.py` | 367-371 | Reset dinâmico em `_reset_server_sync()` |

## Testing

Para testar a implementação:

1. Configure `unlearning_round = 5` em `config.toml`
2. Configure `number_of_clients = 10` e `number_of_rounds = 8`
3. Execute o servidor e clientes
4. Observe os logs para validar:
   - Round 5: Cliente dominante é bloqueado e suas árvores são removidas
   - Servidor aguarda apenas 9 clientes a partir do round 5
   - Métrica de árvores por cliente deve refletir a exclusão

## Considerações Futuras

- Adicionar métricas de desempenho antes/depois do unlearning
- Implementar verificação de integridade do modelo após unlearning
- Adicionar testes unitários para validar comportamento de unlearning
- Considerar implementação de múltiplos clientes para unlearning
