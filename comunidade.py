import praw
import re
import nltk
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import networkx as nx
import community as community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

portuguese_stopwords = nltk.corpus.stopwords.words('portuguese')
portuguese_stopwords.extend(['tcc', 'pra', 'tô', 'aqui', 'lá', 'pro', 'ser', 'ter', 'fazer', 'coisa', 'alguém', 'ainda', 'sobre', 'tudo', 'sei', 'só', 'post', 'trabalho', 'curso', 'pra', 'q', 'vc'])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zà-ú\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("--- Iniciando a Coleta de Dados Aprimorada ---")

client_id = 'SEU_CLIENT_ID'
client_secret = 'SEU_CLIENT_SECRET'
user_agent = 'USER_AGENT'

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

subreddit_name = "faculdadeBR"
search_query = 'TCC'
limit_posts = 50 

subreddit = reddit.subreddit(subreddit_name)

interacoes = [] 
author_texts = defaultdict(str) 

print(f"Coletando até {limit_posts} posts e seus comentários de '{subreddit_name}'...")

for submission in subreddit.search(search_query, limit=limit_posts):
    if submission.author:
        author_texts[submission.author.name] += " " + clean_text(submission.title)
        author_texts[submission.author.name] += " " + clean_text(submission.selftext)

    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        if hasattr(comment, 'author') and comment.author and hasattr(comment, 'parent') and comment.parent().author:
            autor_comentario = comment.author.name
            autor_pai = comment.parent().author.name
            
            if autor_comentario != autor_pai:
                interacoes.append((autor_comentario, autor_pai))
            
            author_texts[autor_comentario] += " " + clean_text(comment.body)

print(f"Coleta finalizada. {len(interacoes)} interações encontradas.")

print("\n--- Construindo a Rede de Interações ---")

G = nx.Graph()
G.add_edges_from(interacoes)

G.remove_nodes_from(list(nx.isolates(G)))

print(f"Rede construída com {G.number_of_nodes()} nós (autores) e {G.number_of_edges()} arestas (interações).")

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw_networkx(G, pos, with_labels=False, node_size=50, width=0.5, node_color='blue', edge_color='gray')
plt.title("Visualização da Rede de Interações do Subreddit")
plt.show()

print("\n--- Detectando e Visualizando as Comunidades na Rede ---")

particao = community_louvain.best_partition(G)
print(f"Detectadas {len(set(particao.values()))} comunidades.")

plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, k=0.15, iterations=20)

cores_dos_nos = [particao.get(node) for node in G.nodes()]

nx.draw_networkx(
    G,
    pos,
    with_labels=False,
    node_size=80,
    width=0.5,
    node_color=cores_dos_nos,
    cmap=plt.cm.get_cmap('viridis') 
)

plt.title("Visualização da Rede com Detecção de Comunidades (Método de Louvain)")
plt.show()

print("\n--- Detectando e Caracterizando as Comunidades na Rede ---")

particao = community_louvain.best_partition(G)

comunidades = defaultdict(list)
for autor, id_comunidade in particao.items():
    comunidades[id_comunidade].append(autor)

print(f"Detectadas {len(comunidades)} comunidades.")

num_comunidades = len(comunidades)
paleta = cm.get_cmap('viridis', num_comunidades)

cores_hex_por_comunidade = {}
for i in range(num_comunidades):
    cor_rgba = paleta(i) 
    cores_hex_por_comunidade[i] = matplotlib.colors.to_hex(cor_rgba) 

for id_comunidade, membros in sorted(comunidades.items()):
    print(f"\n--- Comunidade {id_comunidade} ---")
    print(f"- Cor no Grafo (Hex): {cores_hex_por_comunidade.get(id_comunidade)}")
    print(f"- Total de Membros: {len(membros)}")
    
    limite_membros_para_mostrar = 10 
    
    if len(membros) > limite_membros_para_mostrar:
        print(f"- Amostra de {limite_membros_para_mostrar} membros: {membros[:limite_membros_para_mostrar]}")
    else:
        print(f"- Membros: {membros}")
        
print("\n--- Caracterizando Tópicos por Comunidade ---")

for id_comunidade, membros in sorted(comunidades.items()):
    print(f"\n--- ANÁLISE DA COMUNIDADE {id_comunidade} ---")
    
    membros_com_texto = [autor for autor in membros if autor in author_texts and len(author_texts[autor].strip()) > 0]
    
    print(f"- Total de Membros na Estrutura da Rede: {len(membros)}")
    print(f"- Membros com Texto para Análise: {len(membros_com_texto)}")
    
    texto_comunidade = " ".join([author_texts[autor] for autor in membros_com_texto])
    
    if len(texto_comunidade.strip()) < 100:
        print("- Texto insuficiente para análise temática nesta comunidade.")
        continue

    vectorizer = TfidfVectorizer(
        stop_words=portuguese_stopwords,
        max_features=100,
        ngram_range=(1, 2)
    )
    try:
        tfidf_matrix_comunidade = vectorizer.fit_transform([texto_comunidade])
    
        feature_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix_comunidade.toarray()[0]))
        termos_chave = sorted(feature_scores, key=feature_scores.get, reverse=True)
        
        print("- Termos-Chave da Comunidade:")
        print(f"    {', '.join(termos_chave[:15])}")
    except ValueError:
        print("- Não foi possível vetorizar o texto (corpus muito pequeno ou sem features).")

print("\n--- Fim da Análise de Redes Sociais ---")