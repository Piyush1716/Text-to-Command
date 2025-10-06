import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Load dataset
df = pd.read_csv('c.csv')
df = df[['command', 'category', 'description']]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed descriptions
desc_embeddings = model.encode(df['description'], convert_to_numpy=True)

# Placeholder & description update function
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

# Category keyword mapping
category_keywords = {
    'Navigation': ['where', 'current', 'directory', 'pwd', 'cd'],
    'File Management': ['list', 'create', 'delete', 'copy', 'move', 'mkdir', 'ls', 'rm', 'mv', 'cp', 'touch', 'backup'],
    'Permissions': ['permissions', 'chmod', 'chown', 'apparmor', 'aa-']
}

def get_command(user_query, top_k=3):
    # Optional: filter by category
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
    similarities = np.dot(filtered_embeddings, query_emb) / (np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_emb))
    
    top_idx = similarities.argsort()[-top_k:][::-1]
    results = filtered_df.iloc[top_idx].copy()
    
    # Fill placeholders & update descriptions
    cmds_descs = [fill_placeholders(row['command'], row['description'], user_query) for _, row in results.iterrows()]
    results['command'], results['description'] = zip(*cmds_descs)
    
    return results[['command', 'description', 'category']]

# Test queries
queries = [
    # Navigation
    "where am I right now?",
    "show my current directory",
    "print working directory",
    "go to home folder",
    "change directory to /var/log",
    "navigate to /etc folder",
    "move up one directory",
    "which folder am I in?",
    
    # File Management
    "list all files with details",
    "show hidden files in this folder",
    "I want a detailed view of all files",
    "create a new directory named project",
    "make a folder called backup",
    "delete the folder old_project",
    "copy file.txt to /backup",
    "move oldname.txt to newname.txt",
    "rename myfile.txt to report.txt",
    "create an empty file called notes.txt",
    "remove file temp.log",
    "check file permissions for script.sh",
    
    # Permissions / System
    "change file permissions to read-only",
    "grant execute permission to script.sh",
    "set nginx AppArmor profile to complain mode",
    "enforce security profile for nginx",
    "check who is logged in",
    "view current users on the system",
    "show system run level",
    "enable swap on /dev/sdb1",
    "disable swap on /dev/sdb1",
    "show metadata of encrypted partition /dev/sdb1",
    
    # Miscellaneous / Other
    "show disk usage in this directory",
    "display memory usage",
    "list active processes",
    "search for a file called config.json",
    "find all .txt files in /home/user",
    "show last 10 lines of logfile.log",
    "clear the terminal screen",
    "exit the terminal"
]


for q in queries:
    print(f"\nUser Query: {q}")
    print(get_command(q))
