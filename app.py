import requests
import json
import numpy as np
import faiss
from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer

# Carregar os dados de embeddings
EMBEDDINGS_DIR = "embeddings"
index = faiss.read_index(f"{EMBEDDINGS_DIR}/faiss.index")
all_chunks = np.load(f"{EMBEDDINGS_DIR}/chunks.npy", allow_pickle=True)
chunk_source = np.load(f"{EMBEDDINGS_DIR}/sources.npy", allow_pickle=True)

# Carregar modelo de recuperação
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configuração do Flask
app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>Chatbot de Consulta Offline com Ollama</title>
  </head>
  <body>
    <h1>Chatbot de Consulta aos Documentos</h1>
    <form method="post">
      <input type="text" name="query" size="60" placeholder="Digite sua pergunta">
      <input type="submit" value="Buscar">
    </form>
    {% if answer %}
      <h2>Resposta:</h2>
      <p>{{ answer }}</p>
    {% elif no_answer %}
      <h2>Desculpe, não encontramos informações suficientes.</h2>
      <p>Tente reformular sua pergunta ou forneça mais detalhes.</p>
    {% endif %}
  </body>
</html>
"""

# Função para obter resposta do Ollama via HTTP
def obter_resposta_ollama(query):
    url = "http://localhost:11434/v1/ask"  # Endpoint do Ollama
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [{"role": "user", "content": query}],
        "model": "llama3.2"  # Nome do modelo (ajuste conforme necessário)
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_data = response.json()
        print(f'REQUISIÇÃO DEU CERTO')
        return response_data['text']
    else:
        print(f'FALHOU')
        return "Erro ao chamar o modelo do Ollama."

# Função de recuperação com FAISS
def responder_pergunta(query, top_k=5):
    query_embedding = retrieval_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    # Considerando uma distância alta como baixa similaridade
    if distances[0][0] > 0.5:  # ajuste o limiar conforme necessário
        return None

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": all_chunks[idx],
            "source": chunk_source[idx],
            "distance": distances[0][i]
        })
    return results

# Função que integra recuperação e resposta do Ollama
def gerar_resposta_chatbot(query, top_k=5):
    results = responder_pergunta(query, top_k)
    if results is None:
        return None

    # Montar o contexto usando os resultados recuperados
    contexto = " ".join([f"Fonte: {res['source']} - Trecho: {res['chunk']}" for res in results])
    prompt = f"Baseado nas seguintes informações: {contexto}\nResponda à seguinte pergunta: {query}\nResposta:"

    # Gerar resposta usando o Ollama
    resposta = obter_resposta_ollama(prompt)
    return resposta

# Rota Flask
@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    no_answer = False
    if request.method == "POST":
        query = request.form.get("query", "")
        if query:
            answer = gerar_resposta_chatbot(query)
            if answer is None:
                no_answer = True
    return render_template_string(HTML_TEMPLATE, answer=answer, no_answer=no_answer)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
