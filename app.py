from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import json
import firebase_admin
from firebase_admin import credentials, initialize_app, firestore, auth
import string


app = Flask(__name__)
CORS(app)

# Initialize Firebase
# cred = credentials.Certificate("firebase-key.json")
# firebase_admin.initialize_app(cred)
# db = firestore.client()

# Load firebase key json from environment variable
firebase_key_json = os.getenv('FIREBASE_KEY_JSON')
if not firebase_key_json:
    raise RuntimeError("FIREBASE_KEY_JSON environment variable not set")

cred_dict = json.loads(firebase_key_json)
cred = credentials.Certificate(cred_dict)
initialize_app(cred)
db = firestore.client()

# Load AI model
model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2') slower model but better?


# Cache for journal entries
entries_cache = {}

def load_user_entries(user_id):
    """Load and cache entries for a specific user"""
    if user_id in entries_cache:
        return entries_cache[user_id]
    
    print(f"Loading entries for user {user_id}...")
    try:
        entries_ref = db.collection('journalEntries')\
                       .where('userId', '==', user_id)\
                       .stream()
        
        user_entries = []
        descriptions = []
        
        for doc in entries_ref:
            data = doc.to_dict()
            if 'description' in data:  # Only include entries with descriptions
                user_entries.append({
                    'id': doc.id,
                    'date': data.get('date', ''),
                    'description': data['description'],
                    'mood': data.get('mood'),
                    'imageUrl': data.get('imageUrl')
                })
                descriptions.append(data['description'])
        
        print(f"Loaded {len(user_entries)} entries")
        
        # Only store if we found entries
        if user_entries:
            embeddings = model.encode(descriptions) if descriptions else None
            entries_cache[user_id] = {
                'entries': user_entries,
                'embeddings': embeddings
            }
            return entries_cache[user_id]
            
        return None
        
    except Exception as e:
        print(f"Error loading entries: {str(e)}")
        return None

@app.route('/search')
def search():
    try:
        # Verify user
        id_token = request.headers.get('Authorization')
        if not id_token:
            return jsonify({"error": "Unauthorized"}), 401
            
        decoded_token = auth.verify_id_token(id_token)
        user_id = decoded_token['uid']
        query = request.args.get('q', '').strip().lower()
        
        if not query:
            return jsonify({"error": "Empty query"}), 400
        
        # Load entries - THIS IS WHERE WE ADD DATE HANDLING
        user_data = load_user_entries(user_id)
        if not user_data or not user_data.get('entries'):
            return jsonify({"results": [], "message": "No entries found"})
        
        entries = []
        for entry in user_data['entries']:
            # Standardize the date format for each entry
            raw_date = entry.get('date', '')
            processed_date = ''
            
            # Handle Firestore timestamp
            if hasattr(raw_date, 'timestamp'):  # Firestore timestamp
                processed_date = raw_date.strftime('%Y-%m-%d')
            # Handle string date (YYYY-MM-DD)
            elif isinstance(raw_date, str) and '-' in raw_date:
                processed_date = raw_date
            # Handle datetime object
            elif hasattr(raw_date, 'strftime'):
                processed_date = raw_date.strftime('%Y-%m-%d')
            # Fallback
            else:
                processed_date = str(raw_date) if raw_date else ''
            
            entries.append({
                **entry,
                'date': processed_date  # Use standardized date
            })
        
        embeddings = user_data.get('embeddings')
        
        processed_query = query.translate(str.maketrans('', '', string.punctuation))
        query_words = processed_query.split()
        
        # Find matches (both exact and semantic)
        results = []
        
        # Exact matches (full query or individual words)
        for entry in entries:
            desc = entry['description'].lower()
            exact_score = 0
            
            # Check for full exact match
            if processed_query in desc:
                exact_score += 1.0
                
            # Check for individual word matches
            word_matches = sum(1 for word in query_words if word in desc)
            exact_score += 0.3 * word_matches
            
            if exact_score > 0:
                # Find which specific words matched
                matched_words = [word for word in query_words if word in desc]
                results.append({
                    **entry,
                    'score': exact_score,
                    'match_type': 'exact',
                    'match_details': f"{word_matches} word matches" if word_matches else "full phrase match",
                    'matching_words': matched_words
                })
        
        # Semantic search (if embeddings exist)
        if embeddings is not None and len(embeddings) == len(entries):
            try:
                query_embedding = model.encode(processed_query)
                embeddings_array = np.array(embeddings)
                
                # Calculate document-level similarities
                doc_similarities = util.cos_sim(query_embedding, embeddings_array)
                doc_similarities = doc_similarities[0].cpu().numpy()
                
                for i, doc_score in enumerate(doc_similarities):
                    semantic_score = float(doc_score)
                    if semantic_score > 0.25:  # Entry-level threshold
                        entry_text = entries[i]['description']
                        words = [w for w in entry_text.split() if len(w) > 2]  # Ignore short words
                        
                        if not words:
                            continue
                            
                        # Get embeddings for each word
                        word_embeddings = model.encode(words)
                        
                        # Calculate word-level similarities
                        word_similarities = util.cos_sim(query_embedding, word_embeddings)[0]
                        
                        # Get top matching words
                        matching_words = []
                        for j, word in enumerate(words):
                            word_sim = float(word_similarities[j])
                            if word_sim > 0.35:  # Word-level threshold
                                matching_words.append({
                                    'word': word,
                                    'score': word_sim
                                })
                        
                        # Sort by similarity score
                        matching_words.sort(key=lambda x: x['score'], reverse=True)
                        
                        # Only keep top 5 most relevant words
                        top_matching_words = [w['word'] for w in matching_words[:5]]
                        
                        result = {
                            **entries[i],
                            'score': semantic_score,
                            'match_type': 'semantic',
                            'match_details': "meaning similarity",
                            'matching_words': top_matching_words,
                            'word_similarities': {w['word']: w['score'] for w in matching_words[:3]}  # For debugging
                        }
                        
                        # Update or add to results
                        existing_idx = next((idx for idx, r in enumerate(results) if r['id'] == entries[i]['id']), None)
                        if existing_idx is None or semantic_score > results[existing_idx]['score']:
                            if existing_idx is not None:
                                results[existing_idx] = result
                            else:
                                results.append(result)
            except Exception as e:
                print(f"Semantic search error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Sort results by score (descending) and then by date (newest first)
        results.sort(key=lambda x: (-x['score'], x['date']), reverse=True)

        return jsonify({
            "results": results,  # Return all results
            "query": query,
            "stats": {
                "total_entries": len(entries),
                "exact_matches": sum(1 for r in results if r['match_type'] == 'exact'),
                "semantic_matches": sum(1 for r in results if r['match_type'] == 'semantic')
            }
        })
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=8080, debug=True)