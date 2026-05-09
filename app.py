# app.py — Flask application
from flask import Flask, render_template, request, jsonify
from search import (search_tfidf, search_bm25, search_boolean,
                    search_semantic, search_d2d_tfidf, search_d2d_bm25,
                    search_all, get_stats)

app = Flask(__name__)

@app.route('/')
def index():
    stats = get_stats()
    return render_template('index.html', stats=stats)

@app.route('/api/search')
def search():
    query  = request.args.get('q', '').strip()
    method = request.args.get('method', 'tfidf')
    top_k  = int(request.args.get('k', 10))
    cat    = request.args.get('cat', '')
    year   = request.args.get('year', '')
    mode   = request.args.get('mode', 'AND')  # dùng cho Boolean

    if not query:
        return jsonify({'error': 'Missing query'}), 400

    dispatch = {
        'tfidf':     lambda: search_tfidf(query,     top_k, cat, year),
        'bm25':      lambda: search_bm25(query,      top_k, cat, year),
        'boolean':   lambda: search_boolean(query, mode, top_k, cat, year),
        'semantic':  lambda: search_semantic(query,  top_k, cat, year),
        'd2d_tfidf': lambda: search_d2d_tfidf(query, top_k, cat, year),
        'd2d_bm25':  lambda: search_d2d_bm25(query,  top_k, cat, year),
        'all':       lambda: search_all(query, top_k, cat, year),
    }

    if method not in dispatch:
        return jsonify({'error': f'Invalid method: {method}'}), 400

    result = dispatch[method]()
    result['query'] = query
    return jsonify(result)

@app.route('/api/stats')
def stats():
    return jsonify(get_stats())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)