import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configuration de la page
st.set_page_config(
    page_title="MicroLLM RAG Studio",
    page_icon="🤖",
    layout="wide"
)

# Chemins
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RULES_DIR = BASE_DIR / "rules"
TEMPLATES_DIR = BASE_DIR / "templates"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Créer les dossiers
for dir_path in [DATA_DIR, RULES_DIR, TEMPLATES_DIR, CHROMA_DIR]:
    dir_path.mkdir(exist_ok=True)

# Fichiers par défaut
DEFAULT_DATASET = DATA_DIR / "default.json"
DEFAULT_RULES = RULES_DIR / "default.json"

# ==================== RAG ENGINE ====================

@st.cache_resource
def load_embedding_model():
    """Charge le modèle d'embeddings (mise en cache)"""
    try:
        # Modèle léger et multilingue
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except Exception as e:
        st.error(f"Erreur de chargement du modèle: {e}")
        return None

@st.cache_resource
def get_chroma_client():
    """Initialise ChromaDB (mise en cache)"""
    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        return client
    except Exception as e:
        st.error(f"Erreur ChromaDB: {e}")
        return None

def get_or_create_collection(client):
    """Obtient ou crée la collection ChromaDB"""
    try:
        collection = client.get_collection("knowledge_base")
    except:
        collection = client.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
    return collection

class SimpleRAG:
    """Moteur RAG simplifié pour Streamlit Cloud"""
    
    def __init__(self):
        self.embedder = load_embedding_model()
        self.chroma_client = get_chroma_client()
        
        if self.chroma_client:
            self.collection = get_or_create_collection(self.chroma_client)
        else:
            self.collection = None
    
    def is_ready(self):
        """Vérifie si le RAG est prêt"""
        return self.embedder is not None and self.collection is not None
    
    def add_document(self, question, reponse, metadata=None):
        """Ajoute une paire Q/R à la base vectorielle"""
        if not self.is_ready():
            return False
        
        try:
            # Créer le texte combiné
            text = f"Question: {question}\n\nRéponse: {reponse}"
            
            # Générer l'embedding
            embedding = self.embedder.encode(text).tolist()
            
            # Préparer les métadonnées
            doc_metadata = metadata or {}
            doc_metadata.update({
                "question": question,
                "reponse": reponse,
                "date": datetime.now().isoformat()
            })
            
            # Ajouter à ChromaDB
            doc_id = f"doc_{hash(text)}"
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[doc_metadata],
                ids=[doc_id]
            )
            
            return True
        except Exception as e:
            st.error(f"Erreur lors de l'ajout: {e}")
            return False
    
    def semantic_search(self, query, n_results=5):
        """Recherche sémantique dans la base"""
        if not self.is_ready():
            return []
        
        try:
            # Générer l'embedding de la requête
            query_embedding = self.embedder.encode(query).tolist()
            
            # Rechercher dans ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Formater les résultats
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'question': results['metadatas'][0][i].get('question', ''),
                        'reponse': results['metadatas'][0][i].get('reponse', ''),
                        'categorie': results['metadatas'][0][i].get('categorie', 'N/A'),
                        'score': 1 - results['distances'][0][i],  # Convertir distance en score
                        'distance': results['distances'][0][i]
                    })
            
            return formatted_results
        except Exception as e:
            st.error(f"Erreur de recherche: {e}")
            return []
    
    def get_stats(self):
        """Statistiques de la base"""
        if not self.is_ready():
            return {"total": 0}
        
        try:
            count = self.collection.count()
            return {
                "total": count,
                "embedding_dim": 384,  # Dimension du modèle MiniLM
                "model": "paraphrase-multilingual-MiniLM-L12-v2"
            }
        except:
            return {"total": 0}
    
    def rebuild_index(self, dataset):
        """Reconstruit l'index vectoriel depuis le dataset"""
        if not self.is_ready():
            return False
        
        try:
            # Vider la collection
            self.collection.delete(where={})
            
            # Réindexer tous les documents
            for item in dataset:
                self.add_document(
                    question=item['question'],
                    reponse=item['reponse'],
                    metadata={'categorie': item.get('categorie', 'Général')}
                )
            
            return True
        except Exception as e:
            st.error(f"Erreur lors de la reconstruction: {e}")
            return False
    
    def delete_all(self):
        """Supprime tous les documents"""
        if not self.is_ready():
            return False
        
        try:
            # Supprimer la collection
            self.chroma_client.delete_collection("knowledge_base")
            # Recréer une collection vide
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            st.error(f"Erreur lors de la suppression: {e}")
            return False

# ==================== FONCTIONS UTILITAIRES ====================

def load_json(filepath, default_value=None):
    """Charge un fichier JSON"""
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default_value if default_value is not None else []
    except Exception as e:
        st.error(f"Erreur de chargement {filepath.name}: {str(e)}")
        return default_value if default_value is not None else []

def save_json(filepath, data):
    """Sauvegarde des données en JSON"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Erreur de sauvegarde {filepath.name}: {str(e)}")
        return False

# ==================== INITIALISATION ====================

# Initialiser le RAG
if 'rag_engine' not in st.session_state:
    with st.spinner("🚀 Initialisation du moteur RAG..."):
        st.session_state.rag_engine = SimpleRAG()

# Initialiser les données
if 'dataset' not in st.session_state:
    st.session_state.dataset = load_json(DEFAULT_DATASET, [])

if 'rules' not in st.session_state:
    st.session_state.rules = load_json(DEFAULT_RULES, [])

if 'index_built' not in st.session_state:
    st.session_state.index_built = False

# Construire l'index au premier chargement
if not st.session_state.index_built and len(st.session_state.dataset) > 0:
    with st.spinner("📊 Construction de l'index vectoriel..."):
        if st.session_state.rag_engine.is_ready():
            st.session_state.rag_engine.rebuild_index(st.session_state.dataset)
            st.session_state.index_built = True

# ==================== INTERFACE ====================

# Titre principal
st.title("🤖 MicroLLM RAG Studio")
st.markdown("**Recherche sémantique intelligente** - Propulsé par Sentence Transformers")

# Vérifier l'état du RAG
if not st.session_state.rag_engine.is_ready():
    st.error("⚠️ Le moteur RAG n'a pas pu être initialisé. Certaines fonctionnalités seront limitées.")

# Sidebar
st.sidebar.title("Navigation")

# Afficher les stats dans la sidebar
if st.session_state.rag_engine.is_ready():
    stats = st.session_state.rag_engine.get_stats()
    st.sidebar.metric("📊 Documents indexés", stats['total'])
    st.sidebar.metric("🧠 Dimension embeddings", stats['embedding_dim'])

page = st.sidebar.radio(
    "Choisir une page",
    ["🔍 Recherche Sémantique", "📚 Dataset", "⚙️ Règles", "🎯 Templates", "💾 Export & Maintenance"]
)

st.sidebar.divider()
st.sidebar.markdown("**MicroLLM RAG v2.0**")
st.sidebar.markdown("© 2025 Benjamin Amaad Kama")

# ==================== PAGE: RECHERCHE SÉMANTIQUE ====================

if page == "🔍 Recherche Sémantique":
    st.header("🔍 Recherche Sémantique Intelligente")
    
    if not st.session_state.rag_engine.is_ready():
        st.warning("Le moteur RAG n'est pas disponible. Veuillez vérifier l'installation.")
    elif len(st.session_state.dataset) == 0:
        st.info("💡 Aucune donnée dans la base. Ajoutez des questions/réponses ou chargez un template pour commencer.")
    else:
        st.markdown("""
        La recherche sémantique comprend le **sens** de votre question, pas seulement les mots-clés.
        Essayez différentes formulations de la même question !
        """)
        
        # Zone de recherche
        query = st.text_area(
            "Posez votre question:",
            height=100,
            placeholder="Ex: Comment puis-je prendre des vacances ?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            n_results = st.slider("Nombre de résultats", 1, 10, 5)
        with col2:
            min_score = st.slider("Score minimum", 0.0, 1.0, 0.3, 0.05)
        
        if st.button("🔍 Rechercher", type="primary", use_container_width=True):
            if query:
                with st.spinner("Recherche en cours..."):
                    results = st.session_state.rag_engine.semantic_search(query, n_results=n_results)
                    
                    # Filtrer par score minimum
                    filtered_results = [r for r in results if r['score'] >= min_score]
                    
                    if filtered_results:
                        st.success(f"✅ {len(filtered_results)} résultat(s) trouvé(s)")
                        
                        # Afficher les résultats
                        for i, result in enumerate(filtered_results, 1):
                            score_color = "🟢" if result['score'] > 0.7 else "🟡" if result['score'] > 0.5 else "🟠"
                            
                            with st.expander(
                                f"{score_color} Résultat {i} - Score: {result['score']:.1%} - {result['question'][:60]}...",
                                expanded=(i == 1)
                            ):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**Question similaire:**\n{result['question']}")
                                    st.markdown(f"**Réponse:**\n{result['reponse']}")
                                
                                with col2:
                                    st.metric("Score de pertinence", f"{result['score']:.1%}")
                                    st.metric("Distance", f"{result['distance']:.3f}")
                                    st.info(f"**Catégorie:** {result['categorie']}")
                    else:
                        st.warning(f"❓ Aucun résultat trouvé avec un score supérieur à {min_score:.1%}")
                        st.info("💡 Essayez de reformuler votre question ou de réduire le score minimum.")
            else:
                st.error("❌ Veuillez entrer une question")
        
        # Exemples de recherche
        st.divider()
        st.markdown("### 💡 Exemples de recherche")
        
        example_queries = [
            "Comment réinitialiser mon mot de passe ?",
            "Politique de congés",
            "Horaires d'ouverture",
            "Contactez le support"
        ]
        
        cols = st.columns(len(example_queries))
        for col, example in zip(cols, example_queries):
            with col:
                if st.button(f"📝 {example}", key=f"example_{example}", use_container_width=True):
                    st.rerun()

# ==================== PAGE: DATASET ====================

elif page == "📚 Dataset":
    st.header("Gestion du Dataset")
    
    tab1, tab2, tab3 = st.tabs(["➕ Ajouter", "📋 Liste", "🔄 Import/Export"])
    
    with tab1:
        st.subheader("Ajouter une Question/Réponse")
        
        with st.form("add_qa"):
            question = st.text_input("Question", placeholder="Ex: Comment réinitialiser mon mot de passe ?")
            reponse = st.text_area("Réponse", placeholder="Entrez la réponse détaillée...", height=150)
            categorie = st.selectbox("Catégorie", ["Support", "RH", "Juridique", "Médical", "Général", "Autre"])
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("➕ Ajouter", type="primary", use_container_width=True)
            with col2:
                add_to_index = st.checkbox("Indexer immédiatement", value=True)
            
            if submitted:
                if question and reponse:
                    new_entry = {
                        "id": len(st.session_state.dataset) + 1,
                        "question": question,
                        "reponse": reponse,
                        "categorie": categorie,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Ajouter au dataset
                    st.session_state.dataset.append(new_entry)
                    save_json(DEFAULT_DATASET, st.session_state.dataset)
                    
                    # Ajouter à l'index vectoriel si demandé
                    if add_to_index and st.session_state.rag_engine.is_ready():
                        st.session_state.rag_engine.add_document(
                            question=question,
                            reponse=reponse,
                            metadata={'categorie': categorie}
                        )
                    
                    st.success("✅ Question/Réponse ajoutée avec succès!")
                    st.rerun()
                else:
                    st.error("❌ Veuillez remplir tous les champs")
    
    with tab2:
        st.subheader("Liste des Questions/Réponses")
        
        if st.session_state.dataset:
            # Filtres
            col1, col2 = st.columns([2, 1])
            with col1:
                search_filter = st.text_input("🔍 Filtrer par mots-clés", placeholder="Rechercher dans les questions...")
            with col2:
                category_filter = st.selectbox(
                    "Filtrer par catégorie",
                    ["Toutes"] + list(set([item.get('categorie', 'N/A') for item in st.session_state.dataset]))
                )
            
            # Appliquer les filtres
            filtered_dataset = st.session_state.dataset
            if search_filter:
                filtered_dataset = [
                    item for item in filtered_dataset 
                    if search_filter.lower() in item['question'].lower()
                ]
            if category_filter != "Toutes":
                filtered_dataset = [
                    item for item in filtered_dataset 
                    if item.get('categorie') == category_filter
                ]
            
            st.info(f"📊 {len(filtered_dataset)} / {len(st.session_state.dataset)} entrées affichées")
            
            # Afficher les résultats
            for idx, item in enumerate(filtered_dataset):
                original_idx = st.session_state.dataset.index(item)
                
                with st.expander(f"Q{item.get('id', idx+1)}: {item['question'][:60]}..."):
                    st.markdown(f"**Question:** {item['question']}")
                    st.markdown(f"**Réponse:** {item['reponse']}")
                    st.markdown(f"**Catégorie:** {item.get('categorie', 'N/A')}")
                    st.caption(f"Ajouté le: {item.get('date', 'N/A')}")
                    
                    if st.button(f"🗑️ Supprimer", key=f"del_{original_idx}"):
                        st.session_state.dataset.pop(original_idx)
                        save_json(DEFAULT_DATASET, st.session_state.dataset)
                        st.success("✅ Supprimé! Pensez à reconstruire l'index.")
                        st.rerun()
        else:
            st.info("Aucune question/réponse dans le dataset. Ajoutez-en une!")
    
    with tab3:
        st.subheader("Import/Export Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📤 Exporter**")
            if st.button("Télécharger le dataset"):
                json_str = json.dumps(st.session_state.dataset, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 Télécharger JSON",
                    data=json_str,
                    file_name=f"dataset_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.markdown("**📥 Importer**")
            uploaded_file = st.file_uploader("Importer un fichier JSON", type=['json'])
            if uploaded_file:
                try:
                    data = json.load(uploaded_file)
                    st.session_state.dataset = data
                    save_json(DEFAULT_DATASET, st.session_state.dataset)
                    st.success("✅ Dataset importé avec succès!")
                    st.info("⚠️ Pensez à reconstruire l'index vectoriel dans l'onglet 'Maintenance'")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur d'import: {str(e)}")

# ==================== PAGE: RÈGLES ====================

elif page == "⚙️ Règles":
    st.header("Règles Métiers")
    
    tab1, tab2 = st.tabs(["➕ Ajouter", "📋 Liste"])
    
    with tab1:
        st.subheader("Créer une Règle")
        
        with st.form("add_rule"):
            nom = st.text_input("Nom de la règle", placeholder="Ex: Politesse obligatoire")
            description = st.text_area("Description", placeholder="Décrivez la règle...")
            type_regle = st.selectbox("Type", ["Formatage", "Contenu", "Validation", "Autre"])
            active = st.checkbox("Activer cette règle", value=True)
            
            submitted = st.form_submit_button("➕ Créer la règle")
            
            if submitted:
                if nom and description:
                    new_rule = {
                        "id": len(st.session_state.rules) + 1,
                        "nom": nom,
                        "description": description,
                        "type": type_regle,
                        "active": active,
                        "date": datetime.now().strftime("%Y-%m-