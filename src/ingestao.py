import os
import pickle
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FOLDER = "./src/documentos"
VECTOR_FOLDER = "./src/vetores"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

os.makedirs(VECTOR_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

class RAGDocumentsProcessor():
    def __init__(self, model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
            )
        self.documents = []

    def _extrair_empresa(self, nome_arquivo):
        nome = os.path.splitext(nome_arquivo)[0]
        empresa = nome.split("_")[0]
        return empresa.upper().strip()

    def _load_documents(self):
        for file in os.listdir(PDF_FOLDER):
            if not file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(PDF_FOLDER, file)
            empresa = self._extrair_empresa(file)

            print(f"Lendo {file}")
            print(f"Empresa detectada: {empresa}")

            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if not text.strip():
                    print(f"PDF sem texto: {file}")
                    continue
                chunks = self.splitter.split_text(text)
                for chunk in chunks:
                    embedding_text = f"""
                        Empresa: {empresa}
                        Documento: {file}
                        {chunk}
                        """
                        
                    search_text = f"""
                        {empresa}
                        {file}
                        {chunk}
                        """

                    self.documents.append({
                        "empresa": empresa,
                        "source": file,
                        "text": chunk,
                        "search_text": search_text,
                        "embedding_text": embedding_text
                    })

            except Exception as e:
                print(f"Erro ao processar {file}: {e}")
        print(f"\nTotal chunks: {len(self.documents)}")

    def _generate_embedding(self):
        texts = [ doc["embedding_text"] for doc in self.documents ]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )


        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        index.add(embeddings)

        faiss.write_index(
            index,
            os.path.join(
                VECTOR_FOLDER,
                "faiss.index"
            )
        )

    def _generate_metadata(self):

            metadata = []

            for doc in self.documents:

                metadata.append({
                    "empresa": doc["empresa"],
                    "source": doc["source"],
                    "text": doc["text"],
                    "search_text": doc["search_text"]
                })

            metadata_path = os.path.join(VECTOR_FOLDER, "metadata.pkl")
            with open(metadata_path ,"wb") as f:
                pickle.dump(metadata, f)
                
            print("\nBase vetorial criada com sucesso.")
            print(f"Chunks indexados: {len(metadata)}")
            
    def ingestion(self):
        self._load_documents()
        self._generate_embedding()
        self._generate_metadata()
        
def main():
    processor = RAGDocumentsProcessor()
    processor.ingestion()
    
if __name__ == "__main__":
    main()