import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk 
import re 

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

portuguese_stopwords = nltk.corpus.stopwords.words('portuguese')

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
        'author': submission.author.name if submission.author else 'Unknown',
        'score': submission.score,
        'num_comments': submission.num_comments,
        'created_utc': submission.created_utc,
        'upvote_ratio': submission.upvote_ratio,
        'selftext': submission.selftext,
        'link_flair_text': submission.link_flair_text,
        'url': submission.url 
    })

if not posts_data:
    print("Nenhum post encontrado. Verifique a query, subreddit ou credenciais da API.")
    exit()

df = pd.DataFrame(posts_data)
print(f"{len(df)} posts coletados.")

df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
df['hour_of_day'] = df['created_datetime'].dt.hour
df['day_of_week'] = df['created_datetime'].dt.dayofweek 
df['title_length'] = df['title'].apply(len)
df['selftext_cleaned'] = df['selftext'].apply(clean_text) 
df['selftext_length'] = df['selftext_cleaned'].apply(len)
df['is_selfpost'] = df['url'].str.contains(f'reddit.com/r/{subreddit_name}/comments/')

csv_filename = "posts_faculdade_kdd.csv"
df.to_csv(csv_filename, index=False)
print(f"DataFrame expandido salvo como '{csv_filename}'")
print("\nPrimeiras linhas do DataFrame:")
print(df.head())
print("\nInformações do DataFrame:")
df.info()

print("\n--- 2. Análise Exploratória de Dados (EDA) ---")

print("\nEstatísticas Descritivas (atributos numéricos selecionados):")
numerical_cols_for_stats = ['score', 'num_comments', 'upvote_ratio', 'hour_of_day', 'title_length', 'selftext_length']
print(df[numerical_cols_for_stats].describe())

print("\nVisualizando distribuições...")
plt.figure(figsize=(12, 6))
sns.histplot(df['score'], kde=True, bins=30)
plt.title('Distribuição dos Scores dos Posts')
plt.xlabel('Score')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df['num_comments'], kde=True, bins=30)
plt.title('Distribuição do Número de Comentários')
plt.xlabel('Número de Comentários')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(y=df['num_comments'])
plt.title('Boxplot do Número de Comentários')
plt.ylabel('Número de Comentários')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='day_of_week', palette='viridis')
plt.title('Contagem de Posts por Dia da Semana (0=Seg, 6=Dom)')
plt.xlabel('Dia da Semana')
plt.ylabel('Número de Posts')
plt.tight_layout()
plt.show()

print("\nMatriz de Correlação:")
correlation_matrix = df[numerical_cols_for_stats].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre Atributos Numéricos')
plt.tight_layout()
plt.show()
print(correlation_matrix)

print("\n--- 3. Pré-processamento de Dados ---")

print("\nValores Ausentes por Coluna:")
print(df.isnull().sum())

df['link_flair_text'].fillna('Nenhum', inplace=True)
print("\nValores ausentes em 'link_flair_text' após preenchimento:", df['link_flair_text'].isnull().sum())

features_for_clustering = ['score', 'num_comments', 'title_length', 'selftext_length', 'hour_of_day']
df_clustering = df[features_for_clustering].copy()

df_clustering.dropna(inplace=True)
print(f"\nNúmero de amostras para clustering após remover NaNs: {len(df_clustering)}")

if df_clustering.empty:
    print("Não há dados suficientes para clustering após remover NaNs. Saindo.")
    exit()

print("\nNormalizando/Padronizando dados para clustering...")
scaler = StandardScaler()
df_clustering_scaled = scaler.fit_transform(df_clustering)
df_clustering_scaled = pd.DataFrame(df_clustering_scaled, columns=df_clustering.columns, index=df_clustering.index)
print("Dados escalados (primeiras linhas):")
print(df_clustering_scaled.head())

print("\n--- 5. Aplicação da Técnica de Mineração de Dados (K-Means Clustering) ---")

inertia_values = []
possible_k_values = range(1, 11) 

for k in possible_k_values:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans_temp.fit(df_clustering_scaled)
    inertia_values.append(kmeans_temp.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(possible_k_values, inertia_values, marker='o', linestyle='--')
plt.title('Método do Cotovelo para K-Means')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (Soma das distâncias quadradas intra-cluster)')
plt.xticks(possible_k_values)
plt.grid(True)
plt.tight_layout()
plt.show()

chosen_k = 3 
print(f"Escolhido k = {chosen_k} para K-Means.")

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init='auto')
df.loc[df_clustering_scaled.index, 'cluster'] = kmeans.fit_predict(df_clustering_scaled)

print(f"\nResultados do K-Means (primeiros posts com cluster atribuído):")
print(df.loc[df_clustering_scaled.index, features_for_clustering + ['cluster']].head())

centroids_scaled = kmeans.cluster_centers_
centroids_original_scale = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_original_scale, columns=features_for_clustering)
print("\nCentróides dos Clusters (em escala original):")
print(centroids_df)

print("\nContagem de Posts por Cluster:")
print(df['cluster'].value_counts())

if len(df_clustering_scaled) > 1 and chosen_k > 1: 
    sil_score = silhouette_score(df_clustering_scaled, df.loc[df_clustering_scaled.index, 'cluster'])
    print(f"\nCoeficiente de Silhueta: {sil_score:.3f}")
else:
    print("\nNão foi possível calcular o Coeficiente de Silhueta (poucos clusters ou amostras).")


plt.figure(figsize=(12, 8))
plot_df = df.loc[df_clustering_scaled.index].dropna(subset=['score', 'num_comments', 'cluster'])

sns.scatterplot(data=plot_df,
                x='score',
                y='num_comments',
                hue='cluster',
                palette=sns.color_palette('viridis', n_colors=chosen_k),
                legend='full')
plt.scatter(centroids_df['score'], centroids_df['num_comments'],
            marker='X', s=200, color='red', label='Centróides', edgecolors='black')
plt.title(f'Clusters de Posts (K={chosen_k}) - Score vs Número de Comentários')
plt.xlabel('Score')
plt.ylabel('Número de Comentários')
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Fim da Análise KDD ---")