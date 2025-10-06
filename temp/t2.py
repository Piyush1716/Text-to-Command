import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# --- Load dataset ---
df = pd.read_csv('c.csv')
df = df[['command', 'category', 'description']]

# --- Load embedding model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Precompute description embeddings ---
desc_embeddings = model.encode(df['description'], convert_to_numpy=True)

# --- Helper to fill placeholders ---
def fill_placeholders(command, description, user_query):
    cmd = command
    desc = description
    
    # mkdir
    match = re.search(r'(?:folder|directory) (?:called|named) (\w+)', user_query, re.IGNORECASE)
    if match:
        folder_name = match.group(1)
        if 'mkdir' in cmd:
            cmd = f"mkdir {folder_name}"
            desc = f"Creates a directory named {folder_name}."
    
    # cp
    match = re.search(r'copy (\S+) to (\S+)', user_query, re.IGNORECASE)
    if match and 'cp' in cmd:
        src, dst = match.groups()
        cmd = f"cp {src} {dst}"
        desc = f"Copies {src} to {dst}."
    
    # mv
    match = re.search(r'(?:rename|move) (\S+) to (\S+)', user_query, re.IGNORECASE)
    if match and 'mv' in cmd:
        old, new = match.groups()
        cmd = f"mv {old} {new}"
        desc = f"Renames or moves {old} to {new}."
    
    return cmd, desc

# --- Category keyword mapping ---
category_keywords = {
    'Navigation': ['where', 'current', 'directory', 'pwd', 'cd'],
    'File Management': ['list', 'create', 'delete', 'copy', 'move', 'mkdir', 'ls', 'rm', 'mv', 'cp', 'touch', 'backup'],
    'Permissions': ['permissions', 'chmod', 'chown', 'apparmor', 'aa-']
}

def get_command(user_query, top_k=3):
    # Filter by category
    possible_categories = []
    for cat, keywords in category_keywords.items():
        if any(kw in user_query.lower() for kw in keywords):
            possible_categories.append(cat)
    
    if possible_categories:
        filtered_df = df[df['category'].isin(possible_categories)].copy()
        filtered_embeddings = desc_embeddings[filtered_df.index]
    else:
        filtered_df = df.copy()
        filtered_embeddings = desc_embeddings
    
    # Compute similarity
    query_emb = model.encode(user_query, convert_to_numpy=True)
    similarities = np.dot(filtered_embeddings, query_emb) / (
        np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    
    top_idx = similarities.argsort()[-top_k:][::-1]
    results = filtered_df.iloc[top_idx].copy()
    
    # Fill placeholders & update descriptions
    cmds_descs = [fill_placeholders(row['command'], row['description'], user_query) for _, row in results.iterrows()]
    results['command'], results['description'] = zip(*cmds_descs)
    
    return results[['command', 'description', 'category']]

# --- CLI Loop ---
def main():
    print("Text-to-Command CLI. Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your query: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        results = get_command(user_query)
        print("\nTop matches:")
        for i, row in results.iterrows():
            print(f"Command: {row['command']} | Description: {row['description']} | Category: {row['category']}")

if __name__ == "__main__":
    main()
