## **Project Title**

**Text-to-Command: Natural Language to Linux Command Mapping using NLP**

---

## **Problem Statement**

Many beginner and intermediate Linux users find it challenging to remember the exact syntax of commands. While they may know what they want to achieve, recalling the correct terminal command is difficult. This project aims to bridge the gap by converting natural language queries into the most appropriate Linux commands.

---

## **Aim**

To develop an intelligent system that interprets user queries in plain English and suggests the most relevant Linux terminal commands using NLP and semantic similarity techniques.

---

## **Brief Description**

The project takes **user queries in natural language** (e.g., *"show me hidden files with details"*) and maps them to the closest **Linux command(s)** (e.g., *`ls -la`*).
It uses a **Sentence-BERT model** (`multi-qa-mpnet-base-dot-v1`) to encode both user queries and command descriptions into embeddings and applies **cosine similarity** to find the best match.
A Flask-based frontend allows users to type queries in a web UI and get suggestions in real time.

---

## **NLP Techniques / Models Applied**

* **Sentence Embeddings** using **Sentence-BERT** (`multi-qa-mpnet-base-dot-v1`)
* **Semantic Similarity** with **cosine similarity** (from `sentence-transformers.util`)
* **Top-k Ranking** using **torch.topk** for retrieving multiple candidate commands
* **Query-to-Command Matching** (natural language → structured terminal command)

---

## **Dataset Used**

* **Custom Dataset** created using **Gemini** + manual curation.
* Structure: Each entry contains:

  * **user_query** → natural language variation (e.g., *"how to see current working directory"*)
  * **command** → corresponding Linux command (e.g., *`pwd`*)
  * **description** → short explanation (e.g., *"Prints the current working directory"*)
* Size: ~500+ unique Linux commands, each with 10+ query variations.
* Format: **CSV file** (`commands.csv`)

---

## **Project Output / Deliverable**

1. **Python Module** that returns top-k matching commands for any user query.
2. **Flask-based Web Interface** where users can enter natural language queries and get command suggestions.
3. **Search Results** with ranked commands, original query, matched dataset query, and similarity score.

---

## **Key Features / Unique Contribution**

* Custom-built dataset of Linux command mappings with natural language queries.
* Accurate **semantic search system** for command retrieval.
* **Multi-query support** (handles spelling variations, synonyms, phrasing differences).
* Flask frontend for an **interactive and user-friendly experience**.
* Provides **multiple command suggestions (top-k)** instead of just one.
* Portable — can run on both **Windows and Linux**.

---

## **Real-World Applications**

* **Linux learning tools** for students and beginners.
* **Developer productivity assistant** (reduce time searching for commands).
* **Shell/Terminal integration** as a command recommender.
* **Chatbot integration** (ask in English → get Linux command).
* **DevOps support tool** to quickly recall less frequently used commands.

---
