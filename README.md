# Análise Multifacetada de Comunidades Online no Reddit

Este repositório apresenta uma análise de dados completa de uma comunidade online, demonstrando um pipeline que integra **KDD (Descoberta de Conhecimento em Bases de Dados)**, **Processamento de Linguagem Natural (PLN)** e **Análise de Redes Sociais (SNA)**. A partir de dados do subreddit `r/faculdadeBR`, a análise mapeia a estrutura de interações dos usuários, identifica comunidades e caracteriza o foco temático de cada grupo, revelando as "bolhas" de discussão sobre o tema "TCC".

![Grafo de Comunidades](https://cienciadedatos.net/documentos/pygml02-detecion-comunidades-grafos-redes-python_files/figure-html/unnamed-chunk-5-1.png)
*Visualização do grafo de interações com as comunidades de usuários destacadas por cor.*

## ✨ Habilidades Demonstradas

Este projeto destaca competências práticas nas seguintes áreas de Ciência de Dados:

* **Coleta de Dados via API:**
    * Extração de dados complexos (posts e comentários em árvore) do Reddit utilizando a biblioteca `PRAW`.

* **Análise Exploratória de Dados (EDA):**
    * Manipulação, limpeza e enriquecimento de dados com `Pandas`.
    * Criação de visualizações para análise de distribuição e correlação com `Matplotlib` e `Seaborn`.

* **Machine Learning (Não Supervisionado):**
    * **Clustering:** Aplicação do algoritmo **K-Means** (`Scikit-learn`) para agrupar dados numéricos e textuais.
    * **Detecção de Comunidades:** Uso do **método de Louvain** para identificar grupos em redes complexas.

* **Processamento de Linguagem Natural (PLN):**
    * Técnicas de limpeza e normalização de texto não estruturado com expressões regulares (`re`).
    * Uso de `NLTK` para tratamento de *stopwords*.
    * Vetorização de texto com a técnica **TF-IDF**, incluindo a captura de contexto com n-grams.
    * **Modelagem de Tópicos** para caracterização de corpus textuais.

* **Análise de Redes Sociais (SNA):**
    * Construção, manipulação e análise de grafos de interação social com `NetworkX`.
    * Visualização de redes para apresentar a topologia e as comunidades.

## 📂 Estrutura do Projeto e Jornada Analítica

Os scripts neste repositório representam uma jornada analítica evolutiva, onde cada um aprofunda a investigação sobre o mesmo conjunto de dados, seguindo uma progressão lógica:

* **`1_caracterizacao.py`:** O ponto de partida. Este script foca na **Coleta, Estruturação e Análise Exploratória de Dados (EDA)**. Ele coleta os dados brutos, os organiza em um formato estruturado e realiza a engenharia de features (como `hour_of_day`, `selftext_length`), caracterizando o dataset inicial.

* **`2_kdd.py`:** A primeira abordagem de mineração. Utilizando os dados estruturados e as features criadas no passo anterior, este script aplica o processo de **KDD (Descoberta de Conhecimento em Bases de Dados)** sobre os **metadados** das postagens (score, número de comentários), usando K-Means para encontrar padrões de engajamento.

* **`3_text_mining.py`:** A segunda abordagem foca no **conteúdo textual**. Este script aplica técnicas de PLN para limpar o texto, vetorizá-lo com TF-IDF e, novamente com K-Means, realizar uma **modelagem de tópicos** para descobrir "o que" está sendo discutido.

* **`4_comunidade.py`:** O script final e mais avançado, que integra os conceitos anteriores. Ele eleva a análise para o nível de **rede social**, coletando as **interações** entre usuários. Com `NetworkX` e o método de Louvain, ele identifica as comunidades ("quem" fala com "quem") e as caracteriza tematicamente, respondendo "sobre o que cada comunidade conversa".

## 🚀 Como Executar

Siga os passos abaixo para replicar a análise principal (SNA).

### 1. Pré-requisitos
* Python 3.8 ou superior
* Credenciais de desenvolvedor para a API do Reddit ([link para criar](https://www.reddit.com/prefs/apps))

### 2. Instalação
Clone este repositório e instale as dependências a partir do arquivo `requirements.txt`.
```bash
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio
pip install -r requirements.txt
```
*(**Nota:** Para gerar o arquivo `requirements.txt`, execute `pip freeze > requirements.txt` no seu ambiente virtual).*

### 3. Configuração
No script `comunidade.py` (ou em um arquivo de configuração), insira suas credenciais da API do Reddit:
```python
client_id = 'SEU_CLIENT_ID'
client_secret = 'SEU_CLIENT_SECRET'
user_agent = 'script:analise-comunidades:v1 (by u/SEU_USUARIO_REDDIG)'
```

### 4. Execução
Execute o script principal para rodar o pipeline completo de análise de redes sociais. Os resultados, incluindo a visualização do grafo e a análise textual das comunidades, serão exibidos na saída.
```bash
python comunidade.py
```

## 📈 Principais Achados
* A análise da rede revelou a existência de **múltiplas comunidades** com perfis de interação distintos.
* A caracterização temática mostrou que diferentes comunidades se especializam em diferentes tipos de discussão, como **grupos de apoio emocional**, **nichos de debate técnico** por área de estudo e **redes de ajuda prática**.
* A integração das três abordagens (KDD, PLN e SNA) forneceu uma visão muito mais rica e completa da dinâmica da comunidade do que qualquer uma das técnicas isoladamente.

## 👤 Autor

* **[Luis Felipe Marques Silva]**
* **LinkedIn:** [www.linkedin.com/in/luisfelipemsilva]
* **GitHub:** [www.github.com/Felipao98]
