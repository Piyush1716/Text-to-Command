#!/usr/bin/env python3
"""
nl2cmd_intent_classifier.py

Small prototype for Option B: intent-classification (NL -> base shell command)
with lightweight argument/flag extraction.

Usage:
  - Train and evaluate: python nl2cmd_intent_classifier.py --train
  - Predict single query: python nl2cmd_intent_classifier.py --predict "list all files"
  - Interactive mode: python nl2cmd_intent_classifier.py --interactive

Note: This script only *suggests* commands. It DOES NOT execute anything.
Use it as a starting point and improve dataset, feature extraction, and safety checks.
"""

import re
import os
import argparse
from typing import Optional, Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


MODEL_DIR = "models"
VECT_FILE = os.path.join(MODEL_DIR, "vectorizer.joblib")
MODEL_FILE = os.path.join(MODEL_DIR, "intent_model.joblib")


def make_sample_dataset() -> pd.DataFrame:
    """Return a small starter dataset of (query, label=base_command)."""
    rows = [
        ("list files", "ls"),
        ("show all files", "ls"),
        ("list files in current directory", "ls"),
        ("show hidden files", "ls"),
        ("show hidden files with details", "ls"),
        ("list files with details", "ls"),
        ("print working directory", "pwd"),
        ("what directory am I in", "pwd"),
        ("go to Downloads", "cd"),
        ("change directory to Documents", "cd"),
        ("make directory called test", "mkdir"),
        ("create a folder named backup", "mkdir"),
        ("remove file todo.txt", "rm"),
        ("delete the file report.pdf", "rm"),
        ("delete folder temp recursively", "rm"),
        ("copy file a.txt to b.txt", "cp"),
        ("duplicate file source.txt to dest.txt", "cp"),
        ("move file a.txt to bck/", "mv"),
        ("rename file old.txt to new.txt", "mv"),
        ("show file contents README.md", "cat"),
        ("display contents of log.txt", "cat"),
        ("create empty file notes.txt", "touch"),
        ("search for 'TODO' in files", "grep"),
        ("grep for TODO recursively", "grep"),
        ("find files named config.yaml", "find"),
        ("find large files", "find"),
        ("show disk usage", "du"),
        ("show free disk space", "df"),
        ("list processes", "ps"),
        ("kill process 1234", "kill"),
        ("show git status", "git"),
        ("commit with message 'fix'", "git"),
        ("push to origin main branch", "git"),
        ("download file from url", "wget"),
        ("compress folder logs into logs.tar.gz", "tar"),
        ("extract archive archive.tar.gz", "tar"),
        ("show IP addresses", "ip"),
        ("show network interfaces", "ifconfig"),
    ]
    df = pd.DataFrame(rows, columns=["query", "label"])    
    return df


def preprocess(text: str) -> str:
    t = text.lower().strip()
    # normalize some punctuation and whitespace
    t = re.sub(r"[\"\']", ' ', t)  # remove quotes but keep content
    t = re.sub(r"\s+", ' ', t)
    return t


def train_and_save(df: pd.DataFrame, vectorizer_path: str = VECT_FILE, model_path: str = MODEL_FILE):
    df['query_prep'] = df['query'].apply(preprocess)
    X = df['query_prep'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)

    print("\n=== Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(clf, model_path)
    print(f"Saved vectorizer -> {vectorizer_path}\nSaved model -> {model_path}")


# ---------- Argument / flag extraction helpers ----------

def extract_quoted(text: str) -> Optional[str]:
    m = re.search(r'"(.+?)"|\'(.+?)\'', text)
    if m:
        return m.group(1) or m.group(2)
    return None


def extract_path_or_name(text: str) -> Optional[str]:
    # look for patterns like: in <name>, to <name>, into <name>
    m = re.search(r'\b(?:in|into|inside|to|under|within)\s+([A-Za-z0-9_\-./\\ ]+)', text)
    if m:
        path = m.group(1).strip()
        # cut off common trailing words
        path = re.split(r'\b(recursively|with|and|for|where)\b', path)[0].strip()
        return path
    # fallback: maybe last token is a filename
    tokens = text.strip().split()
    if tokens:
        last = tokens[-1]
        # treat tokens with a dot or starting with / as filename/path
        if '.' in last or last.startswith('/'):
            return last
    return None


def extract_flags(text: str, base_cmd: str) -> str:
    flags = []
    t = text.lower()
    # generic patterns
    if base_cmd == 'ls':
        if re.search(r'\b(hidden|all hidden|hidden files|show hidden|show all)\b', t):
            flags.append('-a')
        if re.search(r'\b(long|details|detailed|with details|list with details|verbose)\b', t):
            flags.append('-l')
        if 'human readable' in t or 'human-readable' in t:
            flags.append('-h')
        if 'recursive' in t:
            flags.append('-R')
    elif base_cmd == 'rm':
        if 'recursive' in t:
            flags.append('-r')
        if 'force' in t or 'without prompt' in t:
            flags.append('-f')
    elif base_cmd in ('cp', 'mv'):
        if 'recursive' in t:
            flags.append('-r')
    elif base_cmd == 'grep':
        if 'ignore case' in t or 'case insensitive' in t:
            flags.append('-i')
        if 'recursive' in t:
            flags.append('-r')
    # add more mappings as you expand commands
    return ' '.join(flags).strip()


def compose_command(base_cmd: str, text: str) -> Tuple[str, dict]:
    """Return (command_string, metadata) without executing it."""
    # try quoted first
    quoted = extract_quoted(text)
    path_or_name = extract_path_or_name(text)
    arg = quoted or path_or_name
    flags = extract_flags(text, base_cmd)

    parts = [base_cmd]
    if flags:
        parts.append(flags)
    if arg:
        parts.append(arg)

    cmd = ' '.join(parts)
    meta = {'base': base_cmd, 'flags': flags, 'arg': arg}
    return cmd, meta


# ---------- Prediction API ----------

def load_model(vectorizer_path: str = VECT_FILE, model_path: str = MODEL_FILE):
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model or vectorizer not found. Run with --train first or provide models.")
    vect = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)
    return vect, clf


def predict_query(text: str, vect, clf) -> dict:
    t = preprocess(text)
    X = vect.transform([t])
    prob = clf.predict_proba(X)[0]
    label = clf.predict(X)[0]
    # get confidence for predicted class
    classes = clf.classes_
    conf = float(prob[list(classes).index(label)])
    cmd, meta = compose_command(label, text)
    return {'query': text, 'predicted_base': label, 'command': cmd, 'confidence': conf, 'meta': meta}


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description='NL -> base-command classifier prototype')
    parser.add_argument('--train', action='store_true', help='Train model on sample dataset (or pass --data CSV)')
    parser.add_argument('--data', type=str, help='Path to CSV dataset with columns query,label (optional)')
    parser.add_argument('--predict', type=str, help='Predict single query (do not execute)')
    parser.add_argument('--interactive', action='store_true', help='Interactive REPL predict mode')
    args = parser.parse_args()

    if args.train:
        if args.data:
            df = pd.read_csv(args.data)
            if 'query' not in df.columns or 'label' not in df.columns:
                raise ValueError('CSV must contain query and label columns')
        else:
            df = make_sample_dataset()
        train_and_save(df)
        return

    if args.predict:
        vect, clf = load_model()
        out = predict_query(args.predict, vect, clf)
        print('\nPredicted base command:', out['predicted_base'])
        print('Confidence:', out['confidence'])
        print('Suggested command (DO NOT EXECUTE):', out['command'])
        print('Meta:', out['meta'])
        return

    if args.interactive:
        vect, clf = load_model()
        print("Interactive mode — type 'quit' or Ctrl-C to exit")
        try:
            while True:
                q = input('> ').strip()
                if not q:
                    continue
                if q.lower() in ('quit', 'exit'):
                    break
                out = predict_query(q, vect, clf)
                print(f"-> {out['command']}   (base={out['predicted_base']}, conf={out['confidence']:.2f})")
        except KeyboardInterrupt:
            print('\nbye')
        return

    parser.print_help()


if __name__ == '__main__':
    main()
