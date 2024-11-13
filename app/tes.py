
import re
import numpy as np
from pymongo import MongoClient
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


client = MongoClient("mongodb://localhost:27017")
db = client["local"]
tf_idf_collection = db['tf_idf']
# feature_collection = db['feature']

# stemmer_factory = StemmerFactory()
# stopword_factory = StopWordRemoverFactory()
# stemmer = stemmer_factory.create_stemmer()
# stopword_remover = stopword_factory.create_stop_word_remover()

# tf_idf_documents = list(tf_idf_collection.find())
tf_idf_documents = list(tf_idf_collection.aggregate([
        {
            "$lookup": {
                "from": "news_data",
                "localField": "Document_id",
                "foreignField": "_id",
                "as": "news_data"
            }
        },
        { "$unwind": "$news_data" },
        { "$project": { "news_data._id": 0, "news_data.Isi Berita": 0, "tfidf_vector": 0 } }
    ]))
# feature_documents = feature_collection.find()

# feature_names = feature_documents[0]['feature_names']
# print(tf_idf_documents)
for document in tf_idf_documents:
    print(document)

# tfidf_vectors = [doc['tfidf_vector'] for doc in tf_idf_documents]
# def inspect_documents(documents, limit=5):
#     """
#     Print the structure of the first few documents
#     """
#     print(f"\nInspecting first {limit} documents:")
#     for i, doc in enumerate(documents[:5000]):
#         print(f"\nDocument {i}:")
#         print("Keys:", doc.keys())
#         print("Types:", {k: type(v) for k, v in doc.items()})
        
# def extract_and_validate_vectors(documents):
#     vectors = []
#     for i, doc in enumerate(documents):
#         try:
#             # Coba akses tfidf_vector
#             vector = doc.get('tfidf_vector')
#             if vector is not None:
#                 # Validasi bahwa vector adalah list/array
#                 if isinstance(vector, (list, tuple)):
#                     vectors.append(vector)
#                 else:
#                     print(f"Warning: Document {i} has invalid vector type: {type(vector)}")
#             else:
#                 print(f"Warning: Document {i} has no tfidf_vector")
#         except Exception as e:
#             print(f"Error processing document {i}: {str(e)}")
            
#         # Opsional: print progress setiap 1000 dokumen
#         if (i + 1) % 1000 == 0:
#             print(f"Processed {i + 1} documents...")
            
#     return vectors

# vectors = extract_and_validate_vectors(tf_idf_documents)