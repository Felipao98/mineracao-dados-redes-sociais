import praw
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

client_id = 'SEU_CLIENT_ID'
client_secret = 'SEU_CLIENT_SECRET'
user_agent = 'USER_AGENT'

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

subreddit = reddit.subreddit("faculdadeBR")
search_query = 'TCC'
limit = 500

posts = []
for submission in subreddit.search(search_query, limit=limit):
    posts.append({
        'id': submission.id,
        'title': submission.title,
        'author': submission.author.name if submission.author else 'Unknown',
        'score': submission.score,
        'num_comments': submission.num_comments
    })


df = pd.DataFrame(posts)
df.to_csv("posts_faculdade.csv", index=False)

G = nx.Graph()

for idx, post in enumerate(posts):
    G.add_node(idx, title=post['title'], score=post['score'], num_comments=post['num_comments'])

for i in range(len(posts)):
    for j in range(i + 1, len(posts)):
        if posts[i]['num_comments'] == posts[j]['num_comments']:
            G.add_edge(i, j)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degree_sequence = [G.degree(n) for n in G.nodes()]
clustering_coeffs = nx.clustering(G)

print(f"Número de vértices: {num_nodes}")
print(f"Número de arestas: {num_edges}")
print("Distribuição de graus (10 primeiros):", degree_sequence[:10])
print("Coeficiente de clustering (10 primeiros):", list(clustering_coeffs.items())[:10])

degree_centrality = nx.degree_centrality(G)

if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(largest_cc)
    eigenvector_centrality = nx.eigenvector_centrality(subG, max_iter=1000)
    eigenvector_centrality = {node: eigenvector_centrality.get(node, 0) for node in G.nodes()}
else:
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

print("Centralidade de grau (nó 0):", degree_centrality.get(0, 0))
print("Centralidade de eigenvector (nó 0):", eigenvector_centrality.get(0, 0))

plt.figure(figsize=(12, 10))

pos = nx.spring_layout(G, seed=42)

node_size = [5000 * degree_centrality.get(node, 0) for node in G.nodes()]
node_color = [eigenvector_centrality.get(node, 0) for node in G.nodes()]

nx.draw(G, pos, with_labels=True, node_size=node_size, node_color=node_color,
        cmap=plt.cm.Blues, font_size=10, font_weight='bold')

plt.title("Rede de Posts - Faculdade / TCC (por número de comentários)")
plt.show()