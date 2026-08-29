# pyre-ignore-all-errors
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any

BANCO_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BANCO_DIR, "historico_exames.db")

def inicializar_banco() -> None:
    """Cria a tabela de histórico de exames se não existir."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historico_exames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT NOT NULL,
            nome_arquivo TEXT NOT NULL,
            modelo_utilizado TEXT NOT NULL,
            classe_id INTEGER NOT NULL,
            diagnostico TEXT NOT NULL,
            confianca REAL NOT NULL,
            prob_saudavel REAL NOT NULL,
            prob_doente REAL NOT NULL,
            tempo_ms REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def salvar_exame(
    nome_arquivo: str,
    modelo_utilizado: str,
    classe_id: int,
    diagnostico: str,
    confianca: float,
    prob_saudavel: float,
    prob_doente: float,
    tempo_ms: float
) -> int:
    """Salva um novo registro de exame processado."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO historico_exames (
            data_hora, nome_arquivo, modelo_utilizado, classe_id,
            diagnostico, confianca, prob_saudavel, prob_doente, tempo_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data_hora, nome_arquivo, modelo_utilizado, classe_id,
        diagnostico, confianca, prob_saudavel, prob_doente, tempo_ms
    ))
    
    registro_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return registro_id

def listar_historico(limite: int = 50) -> List[Dict[str, Any]]:
    """Retorna os últimos exames processados."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, data_hora, nome_arquivo, modelo_utilizado, classe_id,
               diagnostico, confianca, prob_saudavel, prob_doente, tempo_ms
        FROM historico_exames
        ORDER BY id DESC
        LIMIT ?
    """, (limite,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(r) for r in rows]

def limpar_historico() -> None:
    """Limpa todos os registros do histórico."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM historico_exames")
    conn.commit()
    conn.close()

if __name__ == '__main__':
    inicializar_banco()
    print("Banco de dados SQLite inicializado com sucesso em:", DB_PATH)
