import re
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import json

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

with open('data/tf_idf.json', 'r') as file:
    tfidf_data = json.load(file)

with open('data/news_data.json', 'r') as file:
    news_data = json.load(file)
    
with open('data/feature.json', 'r') as file:
    feature_documents = json.load(file)

news_data_dict = {str(item["_id"]["$oid"]): item for item in news_data}

tf_idf_documents = []
for tfidf_item in tfidf_data:
    document_id = str(tfidf_item["Document_id"]["$oid"])
    if document_id in news_data_dict:
        news_item = news_data_dict[document_id]
        
        merged_item = {
            "_id": tfidf_item["_id"],
            "Document_id": tfidf_item["Document_id"],
            "tfidf_vector": tfidf_item["tfidf_vector"],
            "Judul": news_item["Judul"],
            "Tanggal": news_item["Tanggal"],
            "Pengarang": news_item["Pengarang"],
            "Kategori": news_item["Kategori"],
            "Url": news_item["Url"],
            "Slug": news_item["Slug"],
            "Ringkasan": news_item["Ringkasan"]
        }
        tf_idf_documents.append(merged_item)

for i in range(2):
    print(json.dumps(tf_idf_documents[i]))
    
if not tf_idf_documents or not feature_documents:
    raise Exception("No documents found in collections")

print(f"Loaded {len(tf_idf_documents)} TF-IDF documents")
print(f"Loaded {len(feature_documents)} feature documents")

feature_names = feature_documents[0]['feature_names']
vectorizer = TfidfVectorizer(vocabulary=feature_names)
dummy_doc = ' '.join(feature_names)    
vectorizer.fit([dummy_doc])

stemmer_factory = StemmerFactory()
stopword_factory = StopWordRemoverFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_remover = stopword_factory.create_stop_word_remover()

print("Application initialized successfully")
print(f"Vocabulary size: {len(feature_names)}")


def extract_slug(path):
    parts = path.strip('/').split('/')
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return None, parts[-1]
    
def preprocess_text(text):
    text = re.sub(r'[^\w\s]|[\d]', ' ', text)
    text = text.lower()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    tokens = text.split()     
    return tokens

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data["query"]
        category = data.get("category", "").title()        
        
        print(f"Processing search query: {query} for category: {category}")
        
        if category != "All":
            filtered_documents = [doc for doc in tf_idf_documents if doc.get('Kategori') == category]
        else:
            filtered_documents = tf_idf_documents
            
        if not filtered_documents:
            return jsonify({
                "error": f"No documents found for category: {category}"
            }), 404
            
        tfidf_vectors = [doc['tfidf_vector'] for doc in filtered_documents]
        tfidf_matrix_filtered = np.array(tfidf_vectors)
        
        query_tokens = preprocess_text(query)
        query_str = ' '.join(query_tokens)
        query_vector = vectorizer.transform([query_str])
        query_vector_dense = query_vector.toarray().T
        
        scores = np.dot(tfidf_matrix_filtered, query_vector_dense).flatten()
        
        top_k = min(100, len(scores))
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:
                news_data = filtered_documents[idx]
                results.append({
                    "title": news_data.get("Judul", "No title"),
                    "author": news_data.get("Pengarang", "No author"),
                    "summary": news_data.get("Ringkasan", "No summary"),
                    "url": news_data.get("Url", "No Url"),
                    "slug": news_data.get("Slug", "No slug"),
                    "date": news_data.get("Tanggal", "No date"),
                    "category": news_data.get("Kategori", "No category"),
                    "score": float(scores[idx])
                })
        
        print(f"Found {len(results)} results for category: {category}")
        return jsonify(results)
    
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/news/<path:slug>", methods=["GET"])
def get(slug):  
    try:
        category, article_slug = extract_slug(slug)
        for article in news_data:
            if article.get("Slug") == article_slug and article.get("Kategori") == category.title():
                return article 
        if news_data:
            news_data['_id']['$oid'] = str(news_data['_id']['$oid'])
            return jsonify(news_data)
        else:
            return jsonify({"error": "No news data found for the given slug"}), 404
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

@app.route("/", methods=["GET"])
def home():
    return jsonify({    
        "status": "API is running",
        "vocabulary_size": len(feature_names),
        "documents_count": len(tf_idf_documents),
    })

if __name__ == "__main__":
    try:
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Failed to start application: {str(e)}")