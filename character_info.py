import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import spacy

# Constants
VECTOR_DB_PATH = "vector_db"
STORIES_DIR = "stories"

# Initialize Embeddings
mistral = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# CLI Command: compute-embeddings
def compute_embeddings():
    # Check if stories directory exists
    if not os.path.exists(STORIES_DIR):
        print(f"Error: Directory '{STORIES_DIR}' not found.")
        return

    # Read all story files
    story_files = [f for f in os.listdir(STORIES_DIR) if f.endswith(".txt")]
    if not story_files:
        print(f"Error: No story files found in '{STORIES_DIR}'.")
        return

    documents = []
    for file in story_files:
        with open(os.path.join(STORIES_DIR, file), "r", encoding="utf-8") as f:
            story_content = f.read()
            documents.append({"title": file.split(".")[0], "content": story_content})

    # Generate embeddings
    texts = [doc["content"] for doc in documents]
    titles = [doc["title"] for doc in documents]

    # Store embeddings in FAISS vector database
    vector_db = FAISS.from_texts(texts, mistral, metadatas=[{"title": t} for t in titles])
    vector_db.save_local(VECTOR_DB_PATH)
    print("Embeddings computed and saved.")

# CLI Command: get-character-info
def get_character_info(character_name):
    # Load the FAISS vector database
    try:
        vector_db = FAISS.load_local(VECTOR_DB_PATH, mistral, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error: Could not load vector database. {e}")
        return {"error": "Vector database not found. Please run 'compute_embeddings' first."}

    # Search for the character's story
    results = vector_db.similarity_search(character_name, k=1)
    if not results:
        return {"error": "Character not found in the stories."}

    # Extract details from the retrieved story
    story = results[0]
    details = {
        "name": character_name,
        "storyTitle": story.metadata["title"],
        "summary": generate_summary(story.page_content, character_name),
        "relations": extract_relations(story.page_content, character_name),
        "characterType": determine_character_role(story.page_content, character_name),
    }
    return details

# Helper function: Generate a summary of the story
def generate_summary(content, character_name, max_length=50):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    try:
        content_length = len(content)
        max_chunk_size = 1024  # Token limit for the model
        if content_length > max_chunk_size:
            print(f"Story content length ({content_length}) exceeds token limit. Splitting into smaller chunks...")
            chunks = [content[i:i + max_chunk_size] for i in range(0, content_length, max_chunk_size)]
            summaries = [
                summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)[0]["summary_text"]
                for chunk in chunks
            ]
            summary = " ".join(summaries)  # Combine all chunk summaries
        else:
            summary = summarizer(content, max_length=max_length, min_length=30, do_sample=False)[0]["summary_text"]

        # Focus on the character by extracting specific mentions
        sentences = summary.split('. ')
        refined_summary = '. '.join([s for s in sentences if character_name.lower() in s.lower()])
        if not refined_summary:  # Fallback if no mention is found
            refined_summary = '. '.join(sentences[:2]) + ('.' if len(sentences) > 2 else '')
        return refined_summary
    except Exception as e:
        return f"Error in summarization: {e}"

# Helper function: Extract relations
nlp = spacy.load('en_core_web_sm')

def extract_relations(content, character_name):
    doc = nlp(content)
    relations = []

    for sent in doc.sents:
        # Check if the character's name appears in the sentence
        if character_name.lower() in sent.text.lower():
            # Find other PERSON entities in the same sentence
            for ent in sent.ents:
                if ent.label_ == "PERSON" and ent.text.lower() != character_name.lower():
                    # Dynamically describe the relationship using the sentence
                    relation_description = f"Appears in the same context: '{sent.text.strip()}'"
                    relations.append({"name": ent.text, "relation": relation_description})

    # Deduplicate relations
    unique_relations = {rel["name"]: rel for rel in relations}.values()
    return list(unique_relations)


# Helper function: Determine character role
def determine_character_role(content, character_name):
    content_lower = content.lower()
    if character_name.lower() in content_lower:
        if "hero" in content_lower or "protagonist" in content_lower:
            return "Protagonist"
        elif "villain" in content_lower or "antagonist" in content_lower:
            return "Villain"
    return "Side Character"

# Example usage
if __name__ == "__main__":
    command = input("Enter command (compute-embeddings/get-character-info): ").strip()

    if command == "compute-embeddings":
        compute_embeddings()
    elif command == "get-character-info":
        character_name = input("Enter a character name: ").strip()
        character_info = get_character_info(character_name)
        print(character_info)
    else:
        print("Invalid command.")
