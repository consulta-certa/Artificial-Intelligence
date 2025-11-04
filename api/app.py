"""
API Flask SIMPLES para Consulta Certa ML
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
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ==================================================
# CARREGAR MODELOS
# ==================================================

print("🔄 Carregando modelos...")

with open('../models/modelo_noshow.pkl', 'rb') as f:
    modelo_noshow = pickle.load(f)

with open('../models/modelo_clustering.pkl', 'rb') as f:
    modelo_clustering = pickle.load(f)

with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../models/config.json', 'r') as f:
    config = json.load(f)

FEATURES = config['features']
THRESHOLD = config['threshold']

print("✅ Modelos carregados!")

# ==================================================
# FUNÇÕES AUXILIARES
# ==================================================

def conectar_oracle():
    """Conecta ao banco Oracle"""
    return oracledb.connect(
        user=os.getenv('ORACLE_USER'),
        password=os.getenv('ORACLE_PASSWORD'),
        dsn=os.getenv('ORACLE_DSN')
    )


def buscar_dados_saude(id_paciente):
    """Busca dados de saúde do paciente"""
    conn = conectar_oracle()
    cursor = conn.cursor()
    
    query = """
        SELECT idade, genero, tem_hipertensao, tem_diabetes, 
               consome_alcool, possui_deficiencia
        FROM cc_dados_saude_paciente
        WHERE id_paciente = :id_paciente
    """
    
    cursor.execute(query, [id_paciente])
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if not result:
        return None
    
    return {
        'idade': result[0],
        'genero': result[1],
        'tem_hipertensao': result[2],
        'tem_diabetes': result[3],
        'consome_alcool': result[4],
        'possui_deficiencia': result[5]
    }


def verificar_sms_enviado(id_consulta):
    """Verifica se já enviou SMS para a consulta"""
    conn = conectar_oracle()
    cursor = conn.cursor()
    
    query = """
        SELECT COUNT(*) FROM cc_lembretes 
        WHERE id_consulta = :id_consulta AND enviado = 'S'
    """
    
    cursor.execute(query, [id_consulta])
    count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    return count > 0


def salvar_predicao(id_consulta, id_paciente, predicao):
    """Salva predição no banco"""
    conn = conectar_oracle()
    cursor = conn.cursor()
    
    query = """
        INSERT INTO cc_predicoes_noshow 
        (id_consulta, id_paciente, probabilidade_falta, nivel_risco, vai_faltar)
        VALUES (:id_consulta, :id_paciente, :probabilidade, :nivel_risco, :vai_faltar)
    """
    
    cursor.execute(query, {
        'id_consulta': id_consulta,
        'id_paciente': id_paciente,
        'probabilidade': predicao['probabilidade_falta'],
        'nivel_risco': predicao['nivel_risco'],
        'vai_faltar': 'S' if predicao['vai_faltar'] else 'N'
    })
    
    # Atualizar tabela de consultas
    update_query = """
        UPDATE cc_consultas 
        SET risco_noshow = :nivel_risco 
        WHERE id = :id_consulta
    """
    cursor.execute(update_query, {
        'nivel_risco': predicao['nivel_risco'],
        'id_consulta': id_consulta
    })
    
    conn.commit()
    cursor.close()
    conn.close()


def preparar_features(dados_saude, dados_consulta):
    """Prepara features para o modelo"""
    # Converter Oracle (S/N) para modelo (0/1)
    bool_map = {'S': 1, 'N': 0}
    genero_map = {'M': 0, 'F': 1}
    
    # Calcular features temporais
    data_agendamento = datetime.strptime(dados_consulta['data_agendamento'], '%Y-%m-%d').date()
    data_consulta_obj = datetime.strptime(dados_consulta['data_consulta'], '%Y-%m-%d').date()
    
    dias_antecedencia = (data_consulta_obj - data_agendamento).days
    dia_semana = data_consulta_obj.weekday()
    eh_fim_de_semana = 1 if dia_semana >= 5 else 0
    
    # Verificar se já enviou SMS
    sms_received = 1 if verificar_sms_enviado(dados_consulta['id_consulta']) else 0
    
    # Montar features na ORDEM CORRETA
    features = {
        'Gender': genero_map.get(dados_saude['genero'], 0),
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
    
    return features


def calcular_nivel_risco(probabilidade):
    """Calcula nível de risco"""
    if probabilidade >= 0.7:
        return "MUITO_ALTO"
    elif probabilidade >= 0.5:
        return "ALTO"
    elif probabilidade >= 0.3:
        return "MEDIO"
    else:
        return "BAIXO"


def gerar_recomendacoes(nivel_risco):
    """Gera recomendações por nível de risco"""
    recomendacoes = {
        "MUITO_ALTO": {
            "lembretes": 3,
            "canais": ["SMS", "Email", "WhatsApp"],
            "acoes": ["Tutorial em vídeo", "Chatbot proativo", "Ligação telefônica"],
            "prioridade": "CRITICA"
        },
        "ALTO": {
            "lembretes": 2,
            "canais": ["Email", "SMS"],
            "acoes": ["Tutorial passo-a-passo", "Chatbot disponível"],
            "prioridade": "ALTA"
        },
        "MEDIO": {
            "lembretes": 2,
            "canais": ["Email", "SMS"],
            "acoes": ["Lembrete com FAQ"],
            "prioridade": "MEDIA"
        },
        "BAIXO": {
            "lembretes": 1,
            "canais": ["Email"],
            "acoes": ["Lembrete padrão"],
            "prioridade": "BAIXA"
        }
    }
    return recomendacoes.get(nivel_risco, recomendacoes["MEDIO"])


# ==================================================
# ROTAS DA API
# ==================================================

@app.route('/api/ml/health', methods=['GET'])
def health_check():
    """Health check da API"""
    return jsonify({
        "status": "online",
        "modelo": "carregado",
        "versao": config['model_version']
    }), 200


@app.route('/api/ml/predict-noshow', methods=['POST'])
def predict_noshow():
    """
    Prediz risco de no-show
    
    Body JSON:
    {
        "id_consulta": 123,
        "id_paciente": 456,
        "data_agendamento": "2024-11-01",
        "data_consulta": "2024-12-15"
    }
    """
    try:
        data = request.get_json()
        
        # Validar campos obrigatórios
        required = ['id_consulta', 'id_paciente', 'data_agendamento', 'data_consulta']
        for field in required:
            if field not in data:
                return jsonify({"error": f"Campo obrigatório: {field}"}), 400
        
        # Buscar dados de saúde do paciente
        dados_saude = buscar_dados_saude(data['id_paciente'])
        
        if not dados_saude:
            return jsonify({
                "error": "Paciente não preencheu dados de saúde",
                "action": "Redirecionar para questionário"
            }), 404
        
        # Preparar features
        features = preparar_features(dados_saude, data)
        
        # Fazer predição
        X = pd.DataFrame([features])[FEATURES]
        X_scaled = scaler.transform(X)
        
        probabilidade = modelo_noshow.predict_proba(X_scaled)[0][1]
        vai_faltar = probabilidade >= THRESHOLD
        nivel_risco = calcular_nivel_risco(probabilidade)
        recomendacoes = gerar_recomendacoes(nivel_risco)
        
        # Montar resposta
        predicao = {
            "vai_faltar": bool(vai_faltar),
            "probabilidade_falta": round(float(probabilidade), 4),
            "nivel_risco": nivel_risco,
            "recomendacoes": recomendacoes
        }
        
        # Salvar no banco
        salvar_predicao(data['id_consulta'], data['id_paciente'], predicao)
        
        return jsonify({
            "success": True,
            "predicao": predicao,
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ml/dados-saude', methods=['POST'])
def salvar_dados_saude():
    """
    Salva dados de saúde do paciente (primeira vez)
    
    Body JSON:
    {
        "id_paciente": 456,
        "idade": 68,
        "genero": "F",
        "tem_hipertensao": "S",
        "tem_diabetes": "S",
        "consome_alcool": "N",
        "possui_deficiencia": "N",
        "tipo_deficiencia": null
    }
    """
    try:
        data = request.get_json()
        
        conn = conectar_oracle()
        cursor = conn.cursor()
        
        # Verificar se já existe
        check_query = "SELECT id FROM cc_dados_saude_paciente WHERE id_paciente = :id_paciente"
        cursor.execute(check_query, [data['id_paciente']])
        existing = cursor.fetchone()
        
        if existing:
            # UPDATE
            query = """
                UPDATE cc_dados_saude_paciente
                SET idade = :idade, genero = :genero, 
                    tem_hipertensao = :tem_hipertensao, tem_diabetes = :tem_diabetes,
                    consome_alcool = :consome_alcool, possui_deficiencia = :possui_deficiencia,
                    tipo_deficiencia = :tipo_deficiencia
                WHERE id_paciente = :id_paciente
            """
            cursor.execute(query, data)
        else:
            # INSERT
            query = """
                INSERT INTO cc_dados_saude_paciente 
                (id_paciente, idade, genero, tem_hipertensao, tem_diabetes, 
                 consome_alcool, possui_deficiencia, tipo_deficiencia)
                VALUES 
                (:id_paciente, :idade, :genero, :tem_hipertensao, :tem_diabetes,
                 :consome_alcool, :possui_deficiencia, :tipo_deficiencia)
            """
            cursor.execute(query, data)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            "success": True,
            "message": "Dados salvos com sucesso"
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================================================
# EXECUTAR API
# ==================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 CONSULTA CERTA - API ML")
    print("="*60)
    print(f"Versão do modelo: {config['model_version']}")
    print(f"Features: {len(FEATURES)}")
    print(f"Threshold: {THRESHOLD}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)