from flask import Flask, request, jsonify, render_template
import subprocess
import torch
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
import re
import os


app = Flask(__name__)
CORS(app)

def extract_dynamic_values(query):
    """
    Extracts potential filenames or folder names from the user query.
    Returns a list of strings.
    """
    filenames = re.findall(FILENAME_PATTERN, query)
    dirs = re.findall(DIR_PATTERN, query)
    return filenames + dirs


# ----------------------------
# 1. Load saved Sentence-BERT model
# ----------------------------
model = SentenceTransformer("saved_model_2")

# ----------------------------
# 2. Load precomputed embeddings and command list
# ----------------------------
query_embeddings = torch.load("query_embeddings_2.pt")   # tensor of shape [num_commands, embedding_dim]
commands_list = torch.load("commands_list_2.pt")         # list of cmd + description strings in same order

# ----------------------------
# 3. Suggest commands
# ----------------------------

# Regex to detect filenames with common extensions
FILENAME_PATTERN = r'\b[\w\-. ]+\.(txt|sh|log|conf|bin|csv|gz|img|exe)\b'

@app.route('/suggest', methods=['POST'])
def suggest():
    query = request.json.get('query', '')

    # Extract file names from query
    filenames = re.findall(FILENAME_PATTERN, query)

    # Encode query for semantic search
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), query_embeddings)
    topk = torch.topk(scores, k=3)

    suggestions = []

    for score, idx in zip(topk[0], topk[1]):
        cmd = commands_list[idx]

        # Replace common hardcoded filenames in the command with user-provided file
        if filenames:
            file_name = filenames[0]  # take first detected file
            for placeholder in ['file.txt', 'config.conf', 'script.sh', 'error.log', 'access.log']:
                if placeholder in cmd:
                    cmd = cmd.replace(placeholder, file_name)

        suggestions.append({"command": cmd, "score": float(score)})

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
