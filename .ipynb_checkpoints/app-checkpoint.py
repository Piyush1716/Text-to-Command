from flask import Flask, request, jsonify, render_template
import subprocess
import torch
from sentence_transformers import SentenceTransformer
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# ----------------------------
# 1. Load saved Sentence-BERT model
# ----------------------------
model = SentenceTransformer("saved_model")

# ----------------------------
# 2. Load precomputed embeddings and command list
# ----------------------------
query_embeddings = torch.load("query_embeddings.pt")   # tensor of shape [num_commands, embedding_dim]
commands_list = torch.load("commands_list.pt")         # list of cmd + description strings in same order

# ----------------------------
# 3. Suggest commands
# ----------------------------
@app.route('/suggest', methods=['POST'])
def suggest():
    query = request.json.get('query', '')
    query_emb = model.encode(query, convert_to_tensor=True)

    # Cosine similarity
    scores = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), query_embeddings)
    topk = torch.topk(scores, k=3)

    suggestions = []
    for score, idx in zip(topk[0], topk[1]):
        # Split command + description
        if " : " in commands_list[idx]:
            cmd, desc = commands_list[idx].split(" : ", 1)
        else:
            cmd, desc = commands_list[idx], ""
        suggestions.append({"command": cmd, "description": desc, "score": float(score)})
    return jsonify(suggestions)

# ----------------------------
# 4. Execute command safely
# ----------------------------
@app.route('/run', methods=['POST'])
def run_command():
    cmd = request.json.get('command', '')

    # Safety: only allow commands in your dataset
    allowed_cmds = [c.split(" : ")[0] for c in commands_list]
    if cmd not in allowed_cmds:
        return jsonify({"error": "Command not allowed"}), 403

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return jsonify({"stdout": result.stdout, "stderr": result.stderr})
    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------------------
# 5. Serve frontend
# ----------------------------
@app.route('/')
def index():
    return render_template('terminal.html')

# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
