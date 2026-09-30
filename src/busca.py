import pickle
import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

VECTOR_FOLDER = "./src/vetores"
QUERY = "Quais as restrições do Bacen para crédito rural em áreas com risco climático ou desmatamento"

class RAGSearch():
    
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.index = faiss.read_index(f"{VECTOR_FOLDER}/faiss.index")

        with open(f"{VECTOR_FOLDER}/metadata.pkl","rb") as f:
            self.metadata = pickle.load(f)

        self.corpus = [ doc["search_text"].lower() for doc in self.metadata ]

        tokenized_corpus = [ texto.split() for texto in self.corpus ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        
    def _semantic_search(self, query):    
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        distances, indices = self.index.search(query_embedding, k=min(5, len(self.metadata)))
        return distances, indices
        
    def _lexical_search(self, query):
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        return bm25_scores
        
        
    def rank(self, query = QUERY, n = 8):
        distances, indices = self._semantic_search(query)
        lexical_score = self._lexical_search(query)
        resultados = []
        
        for dist, idx in zip(distances[0], indices[0]):
            semantic_score = float(dist)
            keyword_score = lexical_score[idx]
            final_score = (0.7 * semantic_score + 0.3 * keyword_score)
            resultados.append({"idx": idx,"score": final_score})

        resultados.sort(
            key=lambda x: x["score"],
            reverse=True
        )
        resultados = resultados[:n]
        
        if not resultados:
            result = "Nenhum resultado encontrado."
        
        return resultados
    
    def get_texts(self, chunk_list):
        data = []
        for rank, chunk in enumerate(chunk_list, start=1):
            idx = chunk["idx"]
            item = self.metadata[idx]
            texto = item["text"]
            data.append(
                        {"rank":rank,
                        "score":chunk['score'],
                        "arquivo": item["source"],
                        "text": texto}
                        )
            
        result = {"data":data}
        return result
    
    def search(self):
        chunk_list = self.rank()
        text = self.get_texts(chunk_list)
        return text
    
def main():
    searcher = RAGSearch()
    result = searcher.search()
    print(result)
    
if __name__ == "__main__":
    main()
        
