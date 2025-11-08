"""
API Flask para Consulta Certa ML
Versão limpa e comentada: Prediz risco de no-show em consultas médicas usando XGBoost e clustering KMeans.
Integra com banco Oracle para consultar dados de pacientes e salvar predições.
Nota: Dados de saúde são gerenciados pelo front-end; a API apenas consulta e prediz.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, date
import oracledb
import os
import uuid
from dotenv import load_dotenv

# Carregar variáveis de ambiente (ex.: credenciais do banco)
load_dotenv()

# Inicializar Flask e habilitar CORS para integração com front-end
app = Flask(__name__)
CORS(app)

# ==================================================
# CARREGAMENTO DE MODELOS E CONFIGURAÇÕES
# ==================================================
"""
Esta seção carrega os modelos treinados no Colab e configurações.
- modelo_noshow: XGBoost para predizer probabilidade de no-show.
- modelo_clustering: KMeans para agrupar pacientes em clusters (baseado em saúde).
- scaler: Padroniza features para o modelo de no-show (13 features: básicas + clusters).
- scaler_clustering: Padroniza features para clustering (5 features: idade, hipertensão, etc.).
- config: Contém versão, features esperadas, threshold e métricas.
"""

print("🔄 Carregando modelos...")

# Carregar modelo de predição de no-show
with open('./models/modelo_noshow.pkl', 'rb') as f:
    modelo_noshow = pickle.load(f)

# Carregar modelo de clustering
with open('./models/modelo_clustering.pkl', 'rb') as f:
    modelo_clustering = pickle.load(f)

# Carregar scaler para features completas (13)
with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Carregar scaler para features de clustering (5)
with open('./models/scaler_clustering.pkl', 'rb') as f:
    scaler_clustering = pickle.load(f)

# Carregar configurações (features, threshold, etc.)
with open('./models/config.json', 'r') as f:
    config = json.load(f)

# Extrair constantes do config
FEATURES = config['features']  # Lista de 13 features esperadas pelo modelo
THRESHOLD = config['threshold']  # Limite para classificar como "vai faltar"

print("✅ Modelos carregados!")

# ==================================================
# FUNÇÕES AUXILIARES
# ==================================================
"""
Funções de suporte para conectar ao banco, gerar IDs, buscar dados e preparar features.
Essas funções isolam a lógica reutilizável e facilitam manutenção.
"""
 
def buscar_dados_consulta(id_paciente):
    """
    Busca dados da consulta (id_consulta, data_consulta) para o paciente na tabela cc_consultas.
    Assume que há uma consulta "ativa" (ex.: a próxima futura ou a mais recente).
    Retorna um dicionário ou None se não encontrar.
    """
    try:
        conn = conectar_oracle()
        cursor = conn.cursor()
        # Query: Busca a consulta mais próxima futura (ou a mais recente se não houver futura)
        # Ajuste a lógica conforme sua necessidade (ex.: WHERE data_consulta >= SYSDATE ORDER BY data_consulta ASC LIMIT 1)
        query = """
            SELECT id, data_consulta, data_agendamento
            FROM cc_consultas
            WHERE id_paciente = :id_pac
            AND data_consulta >= SYSDATE
            ORDER BY data_consulta ASC
            FETCH FIRST 1 ROW ONLY
        """
        
        cursor.execute(query, {'id_pac': id_paciente})
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            return None
            
        return {
            'id_consulta': result[0],
            'data_consulta': result[1],
            'data_agendamento': result[2]
        }

    except Exception as e:
        print(f"❌ Erro ao buscar dados da consulta: {e}")
        return None

def gerar_uuid():
    """Gera um UUID único compatível com Oracle para IDs de tabelas."""
    return str(uuid.uuid4())

def conectar_oracle():
    """Estabelece conexão com o banco Oracle usando credenciais do .env."""
    return oracledb.connect(
        user=os.getenv('ORACLE_USER'),
        password=os.getenv('ORACLE_PASSWORD'),
        dsn=os.getenv('ORACLE_DSN')
    )

def buscar_dados_saude(id_paciente):
    """
    Busca dados de saúde do paciente no banco Oracle (inseridos pelo front-end).
    Retorna um dicionário com idade, sexo, condições médicas, etc.
    Se não encontrar, retorna None (paciente não preencheu questionário).
    """
    try:
        conn = conectar_oracle()
        cursor = conn.cursor()
        
        query = """
            SELECT idade, sexo, tem_hipertensao, tem_diabetes, 
                   consome_alcool, possui_deficiencia
            FROM cc_dados_saude_paciente
            WHERE id_paciente = :id_pac
        """
        
        cursor.execute(query, {'id_pac': id_paciente})
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not result:
            return None
        
        return {
            'idade': result[0],
            'sexo': result[1],
            'tem_hipertensao': result[2],
            'tem_diabetes': result[3],
            'consome_alcool': result[4],
            'possui_deficiencia': result[5]
        }
    except Exception as e:
        print(f"❌ Erro ao buscar dados de saúde: {e}")
        return None

def verificar_sms_enviado(id_consulta):
    """
    Verifica se já foi enviado SMS para a consulta (usado como feature).
    Retorna True se enviado, False caso contrário.
    """
    try:
        conn = conectar_oracle()
        cursor = conn.cursor()
        
        query = """
            SELECT COUNT(*) FROM cc_lembretes 
            WHERE id_consulta = :id_cons AND enviado = 's'
        """
        
        cursor.execute(query, {'id_cons': id_consulta})
        count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return count > 0
    except Exception as e:
        print(f"❌ Erro ao verificar SMS: {e}")
        return False

def salvar_predicao(id_consulta, id_paciente, predicao):
    """
    Salva a predição no banco Oracle (tabela cc_predicoes_noshow).
    Inclui probabilidade, nível de risco e se vai faltar.
    """
    try:
        conn = conectar_oracle()
        cursor = conn.cursor()
        
        predicao_id = gerar_uuid()
        
        query = """
            INSERT INTO cc_predicoes_noshow 
            (id, id_consulta, id_paciente, probabilidade_falta, nivel_risco, vai_faltar)
            VALUES (:p_id, :p_id_consulta, :p_id_paciente, :p_probabilidade, :p_nivel_risco, :p_vai_faltar)
        """
        
        cursor.execute(query, {
            'p_id': predicao_id,
            'p_id_consulta': id_consulta,
            'p_id_paciente': id_paciente,
            'p_probabilidade': float(predicao['probabilidade_falta']),
            'p_nivel_risco': predicao['nivel_risco'],
            'p_vai_faltar': 's' if predicao['vai_faltar'] else 'n'
        })
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Predição salva: {predicao['nivel_risco']}")
        
    except Exception as e:
        print(f"❌ Erro ao salvar predição: {e}")
        raise

def preparar_features(dados_saude, dados_consulta):
    """
    Prepara as 13 features para o modelo de no-show.
    - Monta 10 features básicas (dados pessoais + temporais).
    - Usa 5 features de saúde para prever cluster com KMeans.
    - Adiciona dummies de cluster (cluster_1, cluster_2, cluster_3).
    Retorna dicionário com todas as features na ordem esperada.
    """
    # Mapeamentos para converter dados do banco (s/n, m/f) em números
    bool_map = {'s': 1, 'n': 0}
    sexo_map = {'m': 0, 'f': 1}
    
    # Calcular features temporais baseadas nas datas
    data_agendamento = dados_consulta['data_agendamento'].date()
    data_consulta_obj = dados_consulta['data_consulta'].date()
    
    dias_antecedencia = (data_consulta_obj - data_agendamento).days
    dia_semana = data_consulta_obj.weekday()
    eh_fim_de_semana = 1 if dia_semana >= 5 else 0
    
    # Verificar se SMS foi enviado (feature adicional)
    sms_received = 1 if verificar_sms_enviado(dados_consulta['id_consulta']) else 0
    
    # Montar 10 features básicas para o modelo final
    features_basicas = {
        'Gender': sexo_map.get(dados_saude['sexo'], 0),
        'Age': dados_saude['idade'],
        'Hipertension': bool_map.get(dados_saude['tem_hipertensao'], 0),
        'Diabetes': bool_map.get(dados_saude['tem_diabetes'], 0),
        'Alcoholism': bool_map.get(dados_saude['consome_alcool'], 0),
        'Handcap': bool_map.get(dados_saude['possui_deficiencia'], 0),
        'SMS_received': sms_received,
        'dias_antecedencia': dias_antecedencia,
        'dia_semana_consulta': dia_semana,
        'eh_fim_de_semana': eh_fim_de_semana
    }
    
    # Montar 5 features específicas para clustering (idade e condições médicas)
    features_clustering = {
        'Age': dados_saude['idade'],
        'Hipertension': bool_map.get(dados_saude['tem_hipertensao'], 0),
        'Diabetes': bool_map.get(dados_saude['tem_diabetes'], 0),
        'Alcoholism': bool_map.get(dados_saude['consome_alcool'], 0),
        'Handcap': bool_map.get(dados_saude['possui_deficiencia'], 0)
    }
    
    # Padronizar as 5 features para clustering e prever cluster
    X_clustering = pd.DataFrame([features_clustering])
    X_clustering_scaled = scaler_clustering.transform(X_clustering)
    cluster_predito = modelo_clustering.predict(X_clustering_scaled)[0]
    
    # Criar dummies binárias para os clusters (0, 1, 2, 3 → cluster_1, cluster_2, cluster_3)
    features_clusters = {
        'cluster_1': 1 if cluster_predito == 1 else 0,
        'cluster_2': 1 if cluster_predito == 2 else 0,
        'cluster_3': 1 if cluster_predito == 3 else 0
    }
    
    # Combinar tudo: 10 básicas + 3 clusters = 13 features
    features = {**features_basicas, **features_clusters}
    
    return features

def calcular_nivel_risco(probabilidade):
    """
    Classifica o nível de risco baseado na probabilidade de falta.
    Retorna string: 'baixo', 'medio', 'alto' ou 'muito_alto'.
    """
    if probabilidade >= 0.7:
        return "muito_alto"
    elif probabilidade >= 0.5:
        return "alto"
    elif probabilidade >= 0.3:
        return "medio"
    else:
        return "baixo"

def gerar_recomendacoes(nivel_risco):
    """
    Gera recomendações personalizadas por nível de risco.
    Inclui lembretes, canais, ações e prioridade.
    """
    recomendacoes = {
        "muito_alto": {
            "lembretes": 3,
            "canais": ["SMS", "Email", "WhatsApp"],
            "acoes": ["Tutorial em vídeo", "Chatbot proativo", "Ligação telefônica"],
            "prioridade": "CRITICA"
        },
        "alto": {
            "lembretes": 2,
            "canais": ["Email", "SMS"],
            "acoes": ["Tutorial passo-a-passo", "Chatbot disponível"],
            "prioridade": "ALTA"
        },
        "medio": {
            "lembretes": 2,
            "canais": ["Email", "SMS"],
            "acoes": ["Lembrete com FAQ"],
            "prioridade": "MEDIA"
        },
        "baixo": {
            "lembretes": 1,
            "canais": ["Email"],
            "acoes": ["Lembrete padrão"],
            "prioridade": "BAIXA"
        }
    }
    return recomendacoes.get(nivel_risco, recomendacoes["medio"])

# ==================================================
# ROTAS DA API
# ==================================================
"""
Rotas REST para interação com o front-end.
- /health: Verifica se a API está online.
- /predict-noshow: Prediz risco de no-show (consulta dados e salva predição).
Nota: Dados de saúde são inseridos pelo front-end diretamente no banco.
"""

@app.route('/api/ml/health', methods=['GET'])
def health_check():
    """Verifica o status da API e retorna versão do modelo."""
    return jsonify({
        "status": "online",
        "modelo": "carregado",
        "versao": config['model_version']
    }), 200

@app.route('/api/ml/predict-noshow', methods=['POST'])
def predict_noshow():
    try:
        data = request.get_json()
        if 'id_paciente' not in data:
            return jsonify({"error": "Campo obrigatório: id_paciente"}), 400
        
        # Buscar dados de saúde
        dados_saude = buscar_dados_saude(data['id_paciente'])
        if not dados_saude:
            return jsonify({"error": "Paciente não preencheu dados de saúde"}), 404
        
        # Buscar dados da consulta
        dados_consulta = buscar_dados_consulta(data['id_paciente'])
        if not dados_consulta:
            return jsonify({"error": "Nenhuma consulta encontrada para o paciente"}), 404
        
        # Preparar features
        features = preparar_features(dados_saude, dados_consulta)
        
        # Predição
        X = pd.DataFrame([features])[FEATURES]
        X_scaled = scaler.transform(X)
        probabilidade = modelo_noshow.predict_proba(X_scaled)[0][1]
        vai_faltar = probabilidade >= THRESHOLD
        nivel_risco = calcular_nivel_risco(probabilidade)
        recomendacoes = gerar_recomendacoes(nivel_risco)
        predicao = {
            "vai_faltar": bool(vai_faltar),
            "probabilidade_falta": round(float(probabilidade), 4),
            "nivel_risco": nivel_risco,
            "recomendacoes": recomendacoes
        }
        
        # Salvar predição
        salvar_predicao(dados_consulta['id_consulta'], data['id_paciente'], predicao)
        return jsonify({"success": True, "predicao": predicao}), 200
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==================================================
# EXECUÇÃO DA API
# ==================================================
"""
Executa a API Flask em modo debug, acessível em localhost:5000.
Use Ctrl+C para parar.
"""

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 CONSULTA CERTA - API ML")
    print("="*60)
    print(f"Versão do modelo: {config['model_version']}")
    print(f"Features: {len(FEATURES)}")
    print(f"Threshold: {THRESHOLD}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
