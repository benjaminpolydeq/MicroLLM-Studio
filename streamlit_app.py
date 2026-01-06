import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="MicroLLM Studio",
    page_icon="🤖",
    layout="wide"
)

# ==================== TRADUCTIONS ====================

TRANSLATIONS = {
    'fr': {
        'title': "🤖 MicroLLM Studio - No Code",
        'subtitle': "Plateforme légère et privée pour créer votre modèle conversationnel",
        'navigation': "Navigation",
        'home': "🏠 Accueil",
        'dataset': "📚 Dataset",
        'rules': "⚙️ Règles",
        'templates': "🎯 Templates",
        'test': "🧪 Test",
        'export': "💾 Export",
        'language': "🌍 Langue",
        
        # Page Accueil
        'welcome': "Bienvenue dans MicroLLM Studio",
        'statistics': "📊 Statistiques",
        'qa_count': "Questions/Réponses",
        'active_rules': "Règles actives",
        'semantic_search': "🧠 Recherche sémantique activée",
        'quick_start': "🚀 Démarrage rapide",
        'quick_start_text': """
        1. Chargez un **template métier** (RH, Support, etc.)
        2. Ajoutez vos **questions/réponses** personnalisées
        3. Définissez des **règles métiers**
        4. **Testez** le modèle
        5. **Exportez** votre configuration
        """,
        'tip': "💡 Conseil: Commencez par charger un template dans l'onglet Templates",
        
        # Page Dataset
        'dataset_management': "Gestion du Dataset",
        'add': "➕ Ajouter",
        'list': "📋 Liste",
        'import_export': "🔄 Import/Export",
        'add_qa': "Ajouter une Question/Réponse",
        'question': "Question",
        'question_placeholder': "Ex: Comment réinitialiser mon mot de passe ?",
        'answer': "Réponse",
        'answer_placeholder': "Entrez la réponse détaillée...",
        'category': "Catégorie",
        'add_button': "➕ Ajouter",
        'success_added': "✅ Question/Réponse ajoutée avec succès!",
        'error_fill_fields': "❌ Veuillez remplir tous les champs",
        'qa_list': "Liste des Questions/Réponses",
        'no_data': "Aucune question/réponse dans le dataset. Ajoutez-en une!",
        'delete': "🗑️ Supprimer",
        'export_data': "📤 Exporter",
        'download_dataset': "Télécharger le dataset",
        'download_json': "💾 Télécharger JSON",
        'import_data': "📥 Importer",
        'import_json': "Importer un fichier JSON",
        'success_import': "✅ Dataset importé avec succès!",
        'error_import': "❌ Erreur d'import:",
        
        # Page Règles
        'rules_management': "Règles Métiers",
        'create_rule': "Créer une Règle",
        'rule_name': "Nom de la règle",
        'rule_name_placeholder': "Ex: Politesse obligatoire",
        'description': "Description",
        'description_placeholder': "Décrivez la règle...",
        'type': "Type",
        'activate_rule': "Activer cette règle",
        'create_rule_button': "➕ Créer la règle",
        'success_rule': "✅ Règle créée avec succès!",
        'rules_list': "Liste des Règles",
        'no_rules': "Aucune règle définie. Créez-en une!",
        'active': "🟢 Active",
        'inactive': "🔴 Inactive",
        'toggle': "🔄 Basculer",
        
        # Page Templates
        'business_templates': "Templates Métiers",
        'template_description': "Choisissez un template prédéfini pour démarrer rapidement",
        'contains': "Contient:",
        'questions_answers': "questions/réponses",
        'rules': "règles",
        'load_template': "⬇️ Charger ce template",
        'success_template': "✅ Template '{name}' chargé avec succès!",
        
        # Page Test
        'test_model': "Tester le Modèle",
        'no_data_warning': "⚠️ Aucune donnée dans le dataset. Ajoutez des questions/réponses d'abord!",
        'ask_question': "Posez une question pour tester le modèle",
        'your_question': "Votre question:",
        'question_input_placeholder': "Tapez votre question ici...",
        'search': "🔍 Rechercher",
        'results_found': "✅ {count} réponse(s) trouvée(s)",
        'similar_question': "**Question similaire:**",
        'answer_label': "**Réponse:**",
        'relevance': "📊 Pertinence:",
        'no_results': "❓ Aucune réponse trouvée. Essayez de reformuler ou ajoutez cette question au dataset.",
        'enter_question': "❌ Veuillez entrer une question",
        
        # Page Export
        'export_save': "Export et Sauvegarde",
        'indexed_docs': "Documents indexés",
        'current_template': "Template actuel:",
        'export_label': "💾 Export",
        'download_rules': "📥 Télécharger Règles",
        'download_all': "📥 Télécharger Tout",
        'auto_save': "🔄 Sauvegarde automatique",
        'auto_save_info': "✅ Toutes vos modifications sont automatiquement sauvegardées localement",
        'advanced_maintenance': "🔧 Maintenance avancée",
        'rebuild_index': "**Reconstruction de l'index de recherche**",
        'rebuild_button': "🔄 Reconstruire l'index",
        'rebuilding': "Reconstruction en cours...",
        'success_rebuild': "✅ Index reconstruit avec succès!",
        'error_rebuild': "❌ Erreur lors de la reconstruction",
        
        # Footer
        'version': "**MicroLLM Studio v1.0**",
        'copyright': "© 2025 Benjamin Amaad Kama",
        'smart_search': "🧠 Recherche intelligente activée",
        'simple_search': "📝 Mode recherche simple",
        
        # Catégories
        'support': "Support",
        'hr': "RH",
        'legal': "Juridique",
        'medical': "Médical",
        'general': "Général",
        'other': "Autre",
        
        # Types de règles
        'formatting': "Formatage",
        'content': "Contenu",
        'validation': "Validation",
    },
    'en': {
        'title': "🤖 MicroLLM Studio - No Code",
        'subtitle': "Lightweight and private platform to create your conversational model",
        'navigation': "Navigation",
        'home': "🏠 Home",
        'dataset': "📚 Dataset",
        'rules': "⚙️ Rules",
        'templates': "🎯 Templates",
        'test': "🧪 Test",
        'export': "💾 Export",
        'language': "🌍 Language",
        
        # Home Page
        'welcome': "Welcome to MicroLLM Studio",
        'statistics': "📊 Statistics",
        'qa_count': "Questions/Answers",
        'active_rules': "Active Rules",
        'semantic_search': "🧠 Semantic search enabled",
        'quick_start': "🚀 Quick Start",
        'quick_start_text': """
        1. Load a **business template** (HR, Support, etc.)
        2. Add your custom **questions/answers**
        3. Define **business rules**
        4. **Test** the model
        5. **Export** your configuration
        """,
        'tip': "💡 Tip: Start by loading a template in the Templates tab",
        
        # Dataset Page
        'dataset_management': "Dataset Management",
        'add': "➕ Add",
        'list': "📋 List",
        'import_export': "🔄 Import/Export",
        'add_qa': "Add a Question/Answer",
        'question': "Question",
        'question_placeholder': "Ex: How to reset my password?",
        'answer': "Answer",
        'answer_placeholder': "Enter the detailed answer...",
        'category': "Category",
        'add_button': "➕ Add",
        'success_added': "✅ Question/Answer added successfully!",
        'error_fill_fields': "❌ Please fill all fields",
        'qa_list': "Questions/Answers List",
        'no_data': "No questions/answers in the dataset. Add one!",
        'delete': "🗑️ Delete",
        'export_data': "📤 Export",
        'download_dataset': "Download dataset",
        'download_json': "💾 Download JSON",
        'import_data': "📥 Import",
        'import_json': "Import a JSON file",
        'success_import': "✅ Dataset imported successfully!",
        'error_import': "❌ Import error:",
        
        # Rules Page
        'rules_management': "Business Rules",
        'create_rule': "Create a Rule",
        'rule_name': "Rule name",
        'rule_name_placeholder': "Ex: Mandatory politeness",
        'description': "Description",
        'description_placeholder': "Describe the rule...",
        'type': "Type",
        'activate_rule': "Activate this rule",
        'create_rule_button': "➕ Create rule",
        'success_rule': "✅ Rule created successfully!",
        'rules_list': "Rules List",
        'no_rules': "No rules defined. Create one!",
        'active': "🟢 Active",
        'inactive': "🔴 Inactive",
        'toggle': "🔄 Toggle",
        
        # Templates Page
        'business_templates': "Business Templates",
        'template_description': "Choose a predefined template to start quickly",
        'contains': "Contains:",
        'questions_answers': "questions/answers",
        'rules': "rules",
        'load_template': "⬇️ Load this template",
        'success_template': "✅ Template '{name}' loaded successfully!",
        
        # Test Page
        'test_model': "Test the Model",
        'no_data_warning': "⚠️ No data in the dataset. Add questions/answers first!",
        'ask_question': "Ask a question to test the model",
        'your_question': "Your question:",
        'question_input_placeholder': "Type your question here...",
        'search': "🔍 Search",
        'results_found': "✅ {count} answer(s) found",
        'similar_question': "**Similar question:**",
        'answer_label': "**Answer:**",
        'relevance': "📊 Relevance:",
        'no_results': "❓ No answer found. Try rephrasing or add this question to the dataset.",
        'enter_question': "❌ Please enter a question",
        
        # Export Page
        'export_save': "Export and Save",
        'indexed_docs': "Indexed documents",
        'current_template': "Current template:",
        'export_label': "💾 Export",
        'download_rules': "📥 Download Rules",
        'download_all': "📥 Download All",
        'auto_save': "🔄 Auto-save",
        'auto_save_info': "✅ All your changes are automatically saved locally",
        'advanced_maintenance': "🔧 Advanced maintenance",
        'rebuild_index': "**Search index reconstruction**",
        'rebuild_button': "🔄 Rebuild index",
        'rebuilding': "Rebuilding...",
        'success_rebuild': "✅ Index rebuilt successfully!",
        'error_rebuild': "❌ Rebuild error",
        
        # Footer
        'version': "**MicroLLM Studio v1.0**",
        'copyright': "© 2025 Benjamin Amaad Kama",
        'smart_search': "🧠 Smart search enabled",
        'simple_search': "📝 Simple search mode",
        
        # Categories
        'support': "Support",
        'hr': "HR",
        'legal': "Legal",
        'medical': "Medical",
        'general': "General",
        'other': "Other",
        
        # Rule types
        'formatting': "Formatting",
        'content': "Content",
        'validation': "Validation",
    }
}

def t(key):
    """Fonction de traduction"""
    lang = st.session_state.get('language', 'fr')
    return TRANSLATIONS[lang].get(key, key)

# ==================== CHEMINS ====================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RULES_DIR = BASE_DIR / "rules"
TEMPLATES_DIR = BASE_DIR / "templates"
EXPORT_DIR = BASE_DIR / "Export"
CHROMA_DIR = BASE_DIR / "chroma_db"

for dir_path in [DATA_DIR, RULES_DIR, TEMPLATES_DIR, EXPORT_DIR, CHROMA_DIR]:
    dir_path.mkdir(exist_ok=True)

DEFAULT_DATASET = DATA_DIR / "default.json"
DEFAULT_RULES = RULES_DIR / "default.json"

# ==================== RAG ENGINE ====================

@st.cache_resource
def load_rag_engine():
    """Charge le moteur RAG en arrière-plan"""
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb.config import Settings
        
        embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        try:
            collection = client.get_collection("knowledge_base")
        except:
            collection = client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
        
        return {
            'embedder': embedder,
            'client': client,
            'collection': collection,
            'enabled': True
        }
    except Exception as e:
        return {'enabled': False, 'error': str(e)}

def load_json(filepath, default_value=None):
    """Charge un fichier JSON"""
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default_value if default_value is not None else []
    except Exception as e:
        st.error(f"Error loading {filepath.name}: {str(e)}")
        return default_value if default_value is not None else []

def save_json(filepath, data):
    """Sauvegarde des données en JSON"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving {filepath.name}: {str(e)}")
        return False

def search_documents(query, dataset, rag_engine, n_results=5):
    """Recherche avec RAG ou fallback"""
    if rag_engine.get('enabled') and rag_engine.get('collection'):
        try:
            embedder = rag_engine['embedder']
            collection = rag_engine['collection']
            
            query_embedding = embedder.encode(query).tolist()
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, len(dataset))
            )
            
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'question': results['metadatas'][0][i].get('question', ''),
                        'reponse': results['metadatas'][0][i].get('reponse', ''),
                        'categorie': results['metadatas'][0][i].get('categorie', 'N/A'),
                        'score': 1 - results['distances'][0][i],
                        'distance': results['distances'][0][i]
                    })
            
            return formatted_results
        except Exception as e:
            pass
    
    # Fallback
    results = []
    query_words = query.lower().split()
    
    for item in dataset:
        question_lower = item['question'].lower()
        matches = sum(1 for word in query_words if word in question_lower)
        if matches > 0:
            results.append({
                'question': item['question'],
                'reponse': item['reponse'],
                'categorie': item.get('categorie', 'N/A'),
                'score': matches / len(query_words),
                'distance': 1 - (matches / len(query_words))
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:n_results]

def add_to_rag_index(question, reponse, metadata, rag_engine):
    """Ajoute un document à l'index RAG"""
    if not rag_engine.get('enabled'):
        return False
    
    try:
        embedder = rag_engine['embedder']
        collection = rag_engine['collection']
        
        text = f"Question: {question}\n\nRéponse: {reponse}"
        embedding = embedder.encode(text).tolist()
        
        doc_metadata = metadata or {}
        doc_metadata.update({
            "question": question,
            "reponse": reponse,
            "date": datetime.now().isoformat()
        })
        
        doc_id = f"doc_{hash(text)}"
        collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[doc_metadata],
            ids=[doc_id]
        )
        
        return True
    except Exception as e:
        return False

def rebuild_rag_index(dataset, rag_engine):
    """Reconstruit l'index RAG"""
    if not rag_engine.get('enabled'):
        return False
    
    try:
        collection = rag_engine['collection']
        
        try:
            collection.delete(where={})
        except:
            pass
        
        for item in dataset:
            add_to_rag_index(
                question=item['question'],
                reponse=item['reponse'],
                metadata={'categorie': item.get('categorie', 'Général')},
                rag_engine=rag_engine
            )
        
        return True
    except Exception as e:
        return False

# ==================== INITIALISATION ====================

if 'language' not in st.session_state:
    st.session_state.language = 'fr'

if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = load_rag_engine()

if 'dataset' not in st.session_state:
    st.session_state.dataset = load_json(DEFAULT_DATASET, [])

if 'rules' not in st.session_state:
    st.session_state.rules = load_json(DEFAULT_RULES, [])

if 'current_template' not in st.session_state:
    st.session_state.current_template = None

if 'index_built' not in st.session_state:
    st.session_state.index_built = False
    if st.session_state.rag_engine.get('enabled') and len(st.session_state.dataset) > 0:
        rebuild_rag_index(st.session_state.dataset, st.session_state.rag_engine)
        st.session_state.index_built = True

# ==================== INTERFACE ====================

# Titre
st.title(t('title'))
st.markdown(t('subtitle'))

# Sidebar
st.sidebar.title(t('navigation'))

# Sélecteur de langue
lang_option = st.sidebar.selectbox(
    t('language'),
    options=['🇫🇷 Français', '🇬🇧 English'],
    index=0 if st.session_state.language == 'fr' else 1
)

new_lang = 'fr' if '🇫🇷' in lang_option else 'en'
if new_lang != st.session_state.language:
    st.session_state.language = new_lang
    st.rerun()

page = st.sidebar.radio(
    "",
    [t('home'), t('dataset'), t('rules'), t('templates'), t('test'), t('export')]
)

# ==================== PAGE: ACCUEIL ====================

if page == t('home'):
    st.header(t('welcome'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t('statistics'))
        st.metric(t('qa_count'), len(st.session_state.dataset))
        st.metric(t('active_rules'), len(st.session_state.rules))
        
        if st.session_state.rag_engine.get('enabled'):
            st.success(t('semantic_search'))
        
    with col2:
        st.subheader(t('quick_start'))
        st.markdown(t('quick_start_text'))
    
    st.info(t('tip'))

# ==================== PAGE: DATASET ====================

elif page == t('dataset'):
    st.header(t('dataset_management'))
    
    tab1, tab2, tab3 = st.tabs([t('add'), t('list'), t('import_export')])
    
    with tab1:
        st.subheader(t('add_qa'))
        
        with st.form("add_qa"):
            question = st.text_input(t('question'), placeholder=t('question_placeholder'))
            reponse = st.text_area(t('answer'), placeholder=t('answer_placeholder'))
            
            categories = [t('support'), t('hr'), t('legal'), t('medical'), t('general'), t('other')]
            categorie = st.selectbox(t('category'), categories)
            
            submitted = st.form_submit_button(t('add_button'))
            
            if submitted:
                if question and reponse:
                    new_entry = {
                        "id": len(st.session_state.dataset) + 1,
                        "question": question,
                        "reponse": reponse,
                        "categorie": categorie,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.dataset.a