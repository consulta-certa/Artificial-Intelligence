Aqui está o README.md completo em uma única caixa de texto para você copiar e colar:

```markdown
# 🤖 Consulta Certa - Sistema de Predição de No-Show

Sistema de Machine Learning para prever e reduzir o absenteísmo em consultas médicas do Hospital das Clínicas.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Oracle](https://img.shields.io/badge/Oracle-Database-red.svg)](https://www.oracle.com/)

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Arquitetura](#arquitetura)
- [Tecnologias](#tecnologias)
- [Instalação](#instalação)
- [Uso](#uso)
- [API Endpoints](#api-endpoints)
- [Modelos de ML](#modelos-de-ml)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Integração com Frontend](#integração-com-frontend)
- [Testes](#testes)
- [Equipe](#equipe)
- [Licença](#licença)

---

## 🎯 Sobre o Projeto

O **Consulta Certa** é uma plataforma digital que visa reduzir o absenteísmo em consultas médicas através de:

- 🔮 **Predição inteligente** de risco de falta
- 📊 **Segmentação de pacientes** por perfil comportamental
- 📱 **Lembretes automáticos** personalizados
- 🤝 **Interface adaptativa** por nível de afinidade digital

### Problema

O Hospital das Clínicas enfrenta uma **taxa de absenteísmo superior a 20%**, resultando em:
- Vagas ociosas
- Desperdício de recursos
- Aumento no tempo de espera
- Prejuízo financeiro estimado em **R$ 150.000/ano**

### Solução

Sistema de Machine Learning que:
1. Prediz a probabilidade de falta em cada consulta
2. Identifica pacientes de alto risco
3. Aciona estratégias preventivas personalizadas
4. Otimiza o envio de lembretes

### Impacto Esperado

- ✅ Redução de **30-40%** nas faltas preveníveis
- ✅ Economia de **R$ 150.000/ano**
- ✅ Melhoria de **40%** na experiência do usuário (NPS)
- ✅ Redução de **30%** nos custos de comunicação

---

## ⚡ Funcionalidades

### 1. Predição de No-Show
- Calcula probabilidade de falta (0-100%)
- Classifica risco: **Baixo**, **Médio**, **Alto**, **Muito Alto**
- Gera recomendações automáticas

### 2. Segmentação de Pacientes
- Agrupa pacientes em perfis de saúde
- Personaliza comunicação e interface
- Otimiza recursos de suporte

### 3. Integração com Sistema Existente
- API REST para integração com frontend React
- Conexão com banco de dados Oracle
- Armazenamento de predições para análise

---

## 🏗️ Arquitetura
```
┌─────────────────┐      HTTP/REST      ┌──────────────────┐
│   FRONTEND      │ ←─────────────────→ │   BACKEND FLASK  │
│  React + TS     │    JSON (API)       │   + ML Models    │
└─────────────────┘                     └──────────────────┘
                                               ↓
                                        ┌──────────────────┐
                                        │  ORACLE DATABASE │
                                        │  + Predições     │
                                        └──────────────────┘
```


### Fluxo de Predição

1. **Paciente agenda consulta** → Sistema coleta dados básicos
2. **API ML recebe request** → Valida campos obrigatórios (`id_consulta`, `id_paciente`)
3. **Busca dados de saúde** → Recupera informações (idade, comorbidades) do banco
4. **Prepara features** → Calcula `dias_antecedencia`, `dia_semana`
5. **Modelo K-Means prediz cluster** → Agrupa paciente por perfil de saúde (5 features)
6. **Modelo XGBoost prediz risco** → Calcula probabilidade de no-show (13 features)
7. **Gera recomendações** → Define estratégias por nível de risco
8. **Salva no banco** → Armazena predição na `cc_predicoes_noshow`
9. **Retorna JSON** → Frontend recebe resultado
10. **Aciona ações** → Sistema envia lembretes personalizados

---

## 🛠️ Tecnologias

### Backend (API)
- **Python 3.8+**
- **Flask** - Framework web
- **Flask-CORS** - Gerenciamento de CORS
- **python-dotenv** - Variáveis de ambiente

### Machine Learning
- **Scikit-Learn** - Modelos de ML
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **XGBoost** - Modelo de classificação

### Banco de Dados
- **Oracle Database** - Armazenamento principal
- **oracledb** - Driver Python para Oracle

### Modelos Treinados
- **XGBoost Classifier** - Predição de no-show
- **K-Means Clustering** - Segmentação de pacientes (usado como feature)

---

## 📦 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- Oracle Database instalado e acessível
- Git

### Passo 1: Clonar o Repositório
```bash
git clone [https://github.com/seu-usuario/consulta-certa-ml.git](https://github.com/seu-usuario/consulta-certa-ml.git)
cd consulta-certa-ml
````

### Passo 2: Criar Ambiente Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Passo 3: Instalar Dependências

```bash
pip install -r requirements.txt
```

**Conteúdo do `requirements.txt`:**

```
flask
flask-cors
pandas
numpy
scikit-learn
xgboost
python-dotenv
oracledb
pickle4
```

### Passo 4: Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# Oracle Database
ORACLE_USER=seu_usuario
ORACLE_PASSWORD=sua_senha
ORACLE_DSN=localhost:1521/XE
```

### Passo 5: Criar Tabelas no Banco

Execute os scripts SQL (veja o `README.md` da outra IA ou o `schema.sql`) para criar as tabelas necessárias:

  - `cc_pacientes`
  - `cc_consultas`
  - `cc_dados_saude_paciente`
  - `cc_predicoes_noshow`
  - `cc_lembretes` (usado pela feature `SMS_received`)

### Passo 6: Baixar Modelos Treinados

1.  Execute o notebook `notebooks/treinamento_modelos.ipynb`
2.  Baixe os 5 arquivos gerados:
      - `modelo_noshow.pkl`
      - `modelo_clustering.pkl`
      - `scaler.pkl`
      - `scaler_clustering.pkl`
      - `config.json`
3.  Coloque-os na pasta `models/`

-----

## 🚀 Uso

### Iniciar a API

```bash
# Navegue até a pasta da API (onde está o app.py)
python app.py
```

**Saída esperada:**

```
🔄 Carregando modelos...
✅ Modelos carregados!

============================================================
🚀 CONSULTA CERTA - API ML
============================================================
Versão do modelo: [versao_do_config.json]
Features: 13
Threshold: [threshold_do_config.json]
============================================================

 * Running on [http://0.0.0.0:5000](http://0.0.0.0:5000)
```

A API estará disponível em: `http://localhost:5000`

### Testar Health Check

```bash
curl http://localhost:5000/api/ml/health
```

**Resposta esperada:**

```json
{
    "status": "online",
    "modelo": "carregado",
    "versao": "[versao_do_config.json]"
}
```

-----

## 📡 API Endpoints

### 1\. Health Check

Verifica se a API está online e os modelos foram carregados.

```http
GET /api/ml/health
```

**Resposta:**

```json
{
    "status": "online",
    "modelo": "carregado",
    "versao": "v1.0.0"
}
```

-----

### 2\. Predição de No-Show

Prediz probabilidade de falta e gera recomendações personalizadas.

```http
POST /api/ml/predict-noshow
Content-Type: application/json
```

**Request Body:**

```json
{
    "id_consulta": "770e8400-e29b-41d4-a716-446655440002",
    "id_paciente": "660e8400-e29b-41d4-a716-446655440001",
    "data_agendamento": "2024-11-01",
    "data_consulta": "2024-12-15"
}
```

**Campos:**

  - `id_consulta` (string, UUID) - ID da consulta agendada
  - `id_paciente` (string, UUID) - ID do paciente
  - `data_agendamento` (string, YYYY-MM-DD) - Data em que foi agendada
  - `data_consulta` (string, YYYY-MM-DD) - Data da consulta

**Resposta de Sucesso (200):**

```json
{
    "success": true,
    "predicao": {
        "vai_faltar": true,
        "probabilidade_falta": 0.7245,
        "nivel_risco": "muito_alto",
        "recomendacoes": {
            "lembretes": 3,
            "canais": ["SMS", "Email", "WhatsApp"],
            "acoes": [
                "Tutorial em vídeo",
                "Chatbot proativo",
                "Ligação telefônica"
            ],
            "prioridade": "CRITICA"
        }
    },
    "timestamp": "2024-11-06T14:30:00"
}
```

**Níveis de Risco:**

  - `baixo` - Probabilidade \< 30%
  - `medio` - Probabilidade 30-50%
  - `alto` - Probabilidade 50-70%
  - `muito_alto` - Probabilidade ≥ 70%

**Resposta de Erro (404):**

```json
{
    "error": "Paciente não preencheu dados de saúde",
    "action": "Redirecionar para questionário"
}
```

-----

## 🤖 Modelos de ML

### Modelo 1: Classificação (XGBoost Classifier)

**Objetivo:** Prever a probabilidade de um paciente faltar à consulta agendada.

**Algoritmo:** XGBoost Classifier (ou similar, via `modelo_noshow.pkl`)

**Features (13):**
O modelo final de predição de no-show utiliza 13 features: 10 features básicas (dados demográficos, de saúde e da consulta) e 3 features de cluster (derivadas do Modelo 2).

| Feature | Descrição | Tipo | Origem |
|---------|-----------|------|--------|
| `Gender` | Gênero | 0=M, 1=F | Dados de saúde |
| `Age` | Idade | Numérico | Dados de saúde |
| `Hipertension` | Tem hipertensão | 0/1 | Dados de saúde |
| `Diabetes` | Tem diabetes | 0/1 | Dados de saúde |
| `Alcoholism` | Consome álcool | 0/1 | Dados de saúde |
| `Handcap` | Possui deficiência | 0/1 | Dados de saúde |
| `SMS_received` | Recebeu lembrete SMS | 0/1 | Tabela `cc_lembretes` |
| `dias_antecedencia` | Dias entre agendamento e consulta | Numérico | Calculado |
| `dia_semana_consulta` | Dia da semana da consulta | 0-6 | Calculado |
| `eh_fim_de_semana` | Consulta é em fim de semana | 0/1 | Calculado |
| `cluster_1` | Paciente pertence ao Cluster 1 | 0/1 | Modelo K-Means |
| `cluster_2` | Paciente pertence ao Cluster 2 | 0/1 | Modelo K-Means |
| `cluster_3` | Paciente pertence ao Cluster 3 | 0/1 | Modelo K-Means |

*(Nota: O Cluster 0 é a base e é representado quando cluster\_1, 2 e 3 são todos 0)*

**Performance (Exemplo):**

  - **ROC-AUC:** \~0.78
  - **Recall:** Otimizado para capturar o máximo de "no-shows" reais.
  - **Threshold otimizado:** Definido em `config.json` (ex: `0.35`) para priorizar o recall.

-----

### Modelo 2: Agrupamento (K-Means Clustering)

**Objetivo:** Segmentar pacientes em grupos com base *apenas* em seu perfil de saúde, para ser usado como feature no modelo principal.

**Algoritmo:** K-Means

**Features (5):**
O modelo de clustering usa 5 features de saúde para criar os perfis.

| Feature | Descrição | Tipo | Origem |
|---------|-----------|------|--------|
| `Age` | Idade | Numérico | Dados de saúde |
| `Hipertension` | Tem hipertensão | 0/1 | Dados de saúde |
| `Diabetes` | Tem diabetes | 0/1 | Dados de saúde |
| `Alcoholism` | Consome álcool | 0/1 | Dados de saúde |
| `Handcap` | Possui deficiência | 0/1 | Dados de saúde |

**Clusters (Perfis de Saúde):**

  - **Cluster 0:** (Base)
  - **Cluster 1:** (Ex: Idosos com comorbidades)
  - **Cluster 2:** (Ex: Jovens com baixo risco de saúde)
  - **Cluster 3:** (Ex: Pacientes com deficiência)

**Uso:** A saída deste modelo (o cluster do paciente) não é usada diretamente. Ela é transformada em 3 features *dummy* (`cluster_1`, `cluster_2`, `cluster_3`) e alimentada ao modelo de XGBoost, melhorando sua capacidade de predição.

-----

## 📁 Estrutura do Projeto

```
/consulta-certa-ml
├── models/
│ ├── modelo_noshow.pkl       # Modelo de Classificação
│ ├── modelo_clustering.pkl   # Modelo K-Means
│ ├── scaler.pkl              # Scaler para modelo principal (13 features)
│ ├── scaler_clustering.pkl   # Scaler para K-Means (5 features)
│ └── config.json             # Configurações (threshold, versão, features)
│
├── app.py                      # Servidor Flask (este código)
├── .env                        # Credenciais do banco (NÃO comitar)
├── .gitignore
├── requirements.txt            # Dependências Python
└── README.md                   # Este arquivo
```

-----

## 🔗 Integração com Frontend (React)

O frontend interage com esta API em dois momentos principais:

**1. No preenchimento do Questionário de Saúde:**
O frontend (React) é responsável por coletar os dados de saúde (idade, gênero, comorbidades) e salvá-los **diretamente** na tabela `cc_dados_saude_paciente`. Esta API de ML *não* faz essa inserção.

**2. Ao Agendar ou Visualizar uma Consulta:**
Este é o fluxo principal:

1.  O Frontend (React) chama `POST /api/ml/predict-noshow` enviando o JSON com os IDs (`id_consulta`, `id_paciente`) e as datas.
2.  A API de ML **consulta** o banco para buscar os dados de saúde (passo 1).
3.  **Cenário de Erro:** Se `buscar_dados_saude` não encontrar o paciente (retornar `None`), a API retorna `404 Not Found` com a mensagem `{"error": "Paciente não preencheu dados de saúde", "action": "Redirecionar para questionário"}`. O frontend deve "capturar" isso e redirecionar o usuário.
4.  **Cenário de Sucesso:** A API roda a predição e retorna `200 OK` com o JSON completo da predição.
5.  O frontend recebe o JSON e usa os campos `nivel_risco` e `recomendacoes` para exibir alertas visuais ou acionar outras lógicas de interface.

-----

## 🧪 Testes

É crucial testar a API simulando o fluxo completo e as dependências do banco de dados. Use o **Insomnia** ou **Postman**.

**Importante:** Para que os testes funcionem, o banco de dados **DEVE** conter os registros "pai" antes da execução.

### Passo 1: Preparar o Banco (SQL)

Antes de testar o `predict-noshow`, você **precisa** ter registros válidos nas 3 tabelas-pai:

1.  Um paciente em `cc_pacientes`
2.  Os dados de saúde desse paciente em `cc_dados_saude_paciente`
3.  A consulta agendada em `cc_consultas`

<!-- end list -->

```sql
-- Exemplo de setup de teste (use UUIDs válidos)
-- 1. Paciente
INSERT INTO cc_pacientes (id, nome, email, ...) VALUES ('paciente-uuid-001', ...);

-- 2. Dados de Saúde
INSERT INTO cc_dados_saude_paciente (id, id_paciente, idade, genero, ...) VALUES ('saude-uuid-001', 'paciente-uuid-001', 50, 'm', ...);

-- 3. Consulta
INSERT INTO cc_consultas (id, especialidade, data_consulta, id_paciente, ...) VALUES ('consulta-uuid-001', 'CARDIOLOGIA', TO_DATE(...), 'paciente-uuid-001', ...);

COMMIT;
```

### Passo 2: Testar `POST /api/ml/predict-noshow` (Insomnia)

Envie uma requisição `POST` para `http://localhost:5000/api/ml/predict-noshow` com o JSON:

```json
{
  "id_consulta": "consulta-uuid-001",
  "id_paciente": "paciente-uuid-001",
  "data_agendamento": "2025-11-01",
  "data_consulta": "2025-11-10"
}
```

**Resultados Esperados:**

1.  **Sucesso:**

      - `Status: 200 OK`
      - Resposta: `{"success": true, "predicao": {...}}`
      - **Verificação no Banco:** Um `SELECT` em `cc_predicoes_noshow` deve mostrar a nova linha de predição.

2.  **Erro (Constraint Violation):**

      - Se você rodar o teste acima *duas vezes*, a segunda falhará.
      - `Status: 500 Internal Server Error`
      - Resposta: `{"success": false, "error": "ORA-00001: unique constraint ... violated"}`
      - **Causa:** A predição para `consulta-uuid-001` já existe. Para testar de novo, delete o registro ou use um `id_consulta` novo (após criá-lo no Passo 1).

3.  **Erro (Paciente sem dados):**

      - Se você usar um `id_paciente` que existe em `cc_pacientes` mas não em `cc_dados_saude_paciente`.
      - `Status: 404 Not Found`
      - Resposta: `{"error": "Paciente não preencheu dados de saúde", ...}`

-----

## 👨‍💻 Equipe

| Nome | Função | Contato |
| :--- | :--- | :--- |
| [Felipe Ferrete] | Desenvolvedor Backend & ML | [https://www.linkedin.com/in/felipe-ferrete-ab63a318a) |
| [Gustavo Bosak] | Desenvolvedor Frontend | [https://www.linkedin.com/in/gustavo-bosak-santos) |
| [Nikolas Brisola] | Desenvolvedor Frontend | [https://www.linkedin.com/in/nikolas-brisola-ab3588353) |

-----

## 📄 Licença

Este projeto é licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

```
```
