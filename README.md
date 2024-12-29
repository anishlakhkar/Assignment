# Character Details Extraction with MistralAI and LangChain

This project extracts structured character details from a dataset of stories using MistralAI embeddings and LangChain. It processes the dataset, computes embeddings, and retrieves information about specific characters in JSON format.

---

## **Features**
- Extract character-specific details, including:
  - Name
  - Story Title
  - Summary
  - Relationships with other characters
  - Character Role (e.g., Protagonist, Villain, Side Character)
- Leverage MistralAI embeddings for processing and FAISS as the vector database.
- Handle edge cases gracefully (e.g., missing data, characters not found).

---

## **Installation**

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Set Up a Python Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## **Directory Structure**
- `stories/`: Contains `.txt` files, each representing a single story.
- `vector_db/`: Stores FAISS embeddings (created during `compute-embeddings`).

Ensure the `stories` directory exists and is populated with `.txt` files before running the script.

---

## **Usage**

Run the script using the following CLI commands:

### 1. **Compute Embeddings**
Processes the dataset, computes embeddings, and saves them in a local FAISS vector database.
```bash
python character_info.py
```
Enter the command:
```bash
compute-embeddings
```

### 2. **Retrieve Character Information**
Search for a specific character and retrieve their details in JSON format.
```bash
python character_info.py
```
Enter the command:
```bash
get-character-info
```
When prompted, input the character's name:
```bash
Enter a character name: <character_name>
```
Example output:
```json
{
  "name": "Jon Snow",
  "storyTitle": "A Song of Ice and Fire",
  "summary": "Jon Snow is a brave and honorable leader who serves as the Lord Commander of the Night's Watch and later unites the Free Folk and Westeros against the threat of the White Walkers.",
  "relations": [
    { "name": "Arya Stark", "relation": "Sister" },
    { "name": "Eddard Stark", "relation": "Father" }
  ],
  "characterType": "Protagonist"
}
```

---

## **Edge Cases**
- **No Stories Found:**
  - Error: "Directory 'stories' not found." Ensure the `stories/` directory exists and contains `.txt` files.
- **Missing Embeddings:**
  - Error: "Vector database not found. Please run 'compute_embeddings' first."
- **Character Not Found:**
  - Returns: `{ "error": "Character not found in the stories." }`

---

## **Requirements**
- Python 3.7+
- MistralAI (via `HuggingFaceEmbeddings`)
- FAISS
- SpaCy
- Transformers

---

## **Troubleshooting**

1. **Missing Dependencies:**
   Ensure all dependencies in `requirements.txt` are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **FAISS Errors:**
   If FAISS fails to load, ensure it is correctly installed and compatible with your Python version.

3. **Unexpected Errors:**
   Check the script output for debugging information. Verify the dataset structure and character name input.

---

## **Future Improvements**
- Enhanced relationship extraction with specific descriptors (e.g., "father," "friend").
- More sophisticated summarization using MistralAI-compatible models.
- Role classification using machine learning.

---

**Author:** Anish Lakhkar
**License:** MIT  

