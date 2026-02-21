"""
Ergotech — Ingestion RAG
Construit le vectorstore au démarrage de l'app (via @st.cache_resource).
Compatible Streamlit Cloud : pas de persistance disque requise.

Usage CLI : python rag/ingest.py  (pour tester localement)
"""

import os
import sys
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import AnthropicEmbeddings

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Stratégie de chunking par sous-répertoire
CHUNK_CONFIG = {
    "reglementation":    {"chunk_size": 800,  "chunk_overlap": 150},
    "evaluation-clinique": {"chunk_size": 600, "chunk_overlap": 120},
    "categories-vph":    {"chunk_size": 500,  "chunk_overlap": 100},
    "modeles-conceptuels": {"chunk_size": 700, "chunk_overlap": 150},
    "argumentaires":     {"chunk_size": 700,  "chunk_overlap": 150},
}
DEFAULT_CHUNK = {"chunk_size": 600, "chunk_overlap": 120}

# ─── Extraction PDF ───────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> List[Document]:
    """Extrait le texte d'un PDF page par page avec métadonnées."""
    docs = []
    try:
        pdf = fitz.open(str(path))
        for page_num, page in enumerate(pdf):
            text = page.get_text().strip()
            if len(text) < 50:  # Ignorer les pages quasi-vides (images)
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_path": str(path.relative_to(BASE_DIR)),
                    "category": path.parent.name,
                    "page": page_num + 1,
                    "total_pages": len(pdf),
                }
            ))
        pdf.close()
        print(f"  📄 {path.name} → {len(docs)} pages extraites")
    except Exception as e:
        print(f"  ⚠️  Erreur sur {path.name} : {e}")
    return docs

# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(raw_docs: List[Document]) -> List[Document]:
    """Découpe les documents en chunks selon leur catégorie."""
    chunks = []
    for doc in raw_docs:
        category = doc.metadata.get("category", "default")
        config = CHUNK_CONFIG.get(category, DEFAULT_CHUNK)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", ".", " "],
        )
        doc_chunks = splitter.split_documents([doc])
        # Conserver les métadonnées dans chaque chunk
        for chunk in doc_chunks:
            chunk.metadata.update(doc.metadata)
        chunks.extend(doc_chunks)
    return chunks

# ─── Ingestion ────────────────────────────────────────────────────────────────

def ingest():
    """Scanne docs/, extrait, chunke et indexe dans ChromaDB."""
    print("\n🚀 Ergotech RAG — Ingestion\n")

    # Vérifier la clé API
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY manquante. Créez un fichier .env avec votre clé.")
        sys.exit(1)

    # Scanner tous les PDFs
    pdf_files = sorted(DOCS_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"❌ Aucun PDF trouvé dans {DOCS_DIR}")
        sys.exit(1)

    print(f"📂 {len(pdf_files)} document(s) trouvé(s) :\n")
    for f in pdf_files:
        print(f"  • {f.relative_to(BASE_DIR)}")
    print()

    # Extraction
    print("📖 Extraction du texte...\n")
    all_raw = []
    for pdf_path in pdf_files:
        all_raw.extend(extract_pdf(pdf_path))

    print(f"\n✅ {len(all_raw)} pages extraites au total\n")

    # Chunking
    print("✂️  Découpage en chunks...\n")
    all_chunks = chunk_documents(all_raw)
    print(f"✅ {len(all_chunks)} chunks générés\n")

    # Stats par catégorie
    from collections import Counter
    cat_counts = Counter(c.metadata["category"] for c in all_chunks)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:<25} {count} chunks")
    print()

    # Vectorisation et indexation
    print("🔢 Vectorisation et indexation dans ChromaDB...\n")
    embeddings = AnthropicEmbeddings(model="voyage-3")

    VECTORSTORE_DIR.mkdir(exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name="ergotech",
    )

    print(f"✅ {vectorstore._collection.count()} vecteurs indexés")
    print(f"📦 Vectorstore sauvegardé dans : {VECTORSTORE_DIR}\n")
    print("🎉 Ingestion terminée ! Le RAG est prêt.\n")

if __name__ == "__main__":
    # Charger .env si présent
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass
    ingest()
