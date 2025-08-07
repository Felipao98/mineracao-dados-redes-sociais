import praw
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk 
import re
from wordcloud import WordCloud

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

portuguese_stopwords = nltk.corpus.stopwords.words('portuguese')
portuguese_stopwords.extend(['tcc', 'pra', 'tô', 'aqui', 'lá', 'pro', 'ser', 'ter', 'fazer', 'coisa', 'alguém', 'ainda', 'sobre', 'tudo', 'sei', 'só', 'post', 'trabalho', 'curso'])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() 
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'[^a-zà-ú\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

print("--- 1. Coleta de Dados e Construção da Base Estruturada ---")
client_id = 'SEU_CLIENT_ID'
client_secret = 'SEU_CLIENT_SECRET'
user_agent = 'USER_AGENT'

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

subreddit_name = "faculdadeBR"
search_query = 'TCC'
limit_posts = 150 

subreddit = reddit.subreddit(subreddit_name)

posts_data = []
print(f"Coletando até {limit_posts} posts do subreddit '{subreddit_name}' com a query '{search_query}'...")
for submission in subreddit.search(search_query, limit=limit_posts):
    posts_data.append({
        'id': submission.id,
        'title': submission.title,
        'selftext': submission.selftext
    })

if not posts_data:
    print("Nenhum post encontrado. Verifique a query, subreddit ou credenciais da API.")
    exit()

df = pd.DataFrame(posts_data)
print(f"{len(df)} posts coletados.")

print("\n--- 2. Pré-processamento e Análise Exploratória de Texto ---")

df['selftext_cleaned'] = df['selftext'].apply(clean_text)

df = df[df['selftext_cleaned'].str.len() > 10].copy() 
print(f"Número de posts restantes após limpeza: {len(df)}")

print("\nGerando nuvem de palavras a partir de todos os posts...")
all_text = " ".join(review for review in df.selftext_cleaned)
wordcloud = WordCloud(stopwords=portuguese_stopwords,
                      background_color="white",
                      width=800,
                      height=400,
                      colormap='viridis').generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nuvem de Palavras Mais Comuns nos Posts sobre TCC")
plt.tight_layout()
plt.show()

print("\n--- 3. Transformação de Texto em Vetores com TF-IDF ---")
print("Este passo converte o texto limpo em uma representação numérica que a máquina pode entender.")

vectorizer = TfidfVectorizer(
    stop_words=portuguese_stopwords,
    max_features=1000, 
    ngram_range=(1, 2)  
)

tfidf_matrix = vectorizer.fit_transform(df['selftext_cleaned'])

print("Dimensões da matriz TF-IDF:", tfidf_matrix.shape)
print("(Linhas = número de posts, Colunas = número de palavras únicas/features)")

print("\n--- 4. Mineração de Texto: Modelagem de Tópicos com K-Means ---")
print("Agrupando os posts em clusters com base no conteúdo textual (vetores TF-IDF).")

inertia_values = []
possible_k_values = range(2, 11) 

for k in possible_k_values:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans_temp.fit(tfidf_matrix)
    inertia_values.append(kmeans_temp.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(possible_k_values, inertia_values, marker='o', linestyle='--')
plt.title('Método do Cotovelo para K-Means (em dados de texto)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.xticks(possible_k_values)
plt.grid(True)
plt.tight_layout()
plt.show()

chosen_k = 4 
print(f"Escolhido k = {chosen_k} para a modelagem de tópicos.")

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

print("\nContagem de Posts por Cluster (Tópico):")
print(df['cluster'].value_counts())

print("\n--- 5. Análise dos Tópicos Encontrados ---")
print("Analisando as palavras mais significativas de cada cluster para entender o tópico.")

terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(chosen_k):
    print(f"\nTópico {i}:")
    for ind in order_centroids[i, :10]:
        print(f' {terms[ind]}')

print("\nExemplo de posts de cada tópico:")
for i in range(chosen_k):
    print(f"\n--- Tópico {i} ---")
    
    num_exemplos = 3
    sample_posts = df[df['cluster'] == i].sample(n=min(num_exemplos, len(df[df['cluster'] == i])), random_state=1)
    
    for index, row in sample_posts.iterrows():
        print(f"Post ID: {row['id']}\nTítulo: {row['title']}\nTexto: {row['selftext'][:200]}...\\n")


print("\n--- Fim da Análise de Mineração de Texto ---")