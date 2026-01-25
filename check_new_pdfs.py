from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from populate_database import load_documents, split_documents, calculate_chunk_ids

# Load existing database
db = Chroma(persist_directory="chroma", embedding_function=get_embedding_function())
existing_items = db.get(include=[])
existing_ids = set(existing_items["ids"])

# Get existing PDF sources from database
existing_sources = set([id.split(':')[0] for id in existing_ids])
print(f"Existing PDFs in database: {len(existing_sources)}")

# Load current PDFs
docs = load_documents()
chunks = split_documents(docs)
chunks_with_ids = calculate_chunk_ids(chunks)

# Get current PDF sources
current_sources = set([chunk.metadata.get("source") for chunk in chunks_with_ids])
print(f"Current PDFs in data folder: {len(current_sources)}")

# Find PDFs in database but not in current folder
missing_in_folder = existing_sources - current_sources
print(f"\nPDFs in database but not in current folder: {len(missing_in_folder)}")
for source in sorted(missing_in_folder):
    print(f"  - {source}")

# Find PDFs in current folder but not in database
new_pdfs = current_sources - existing_sources
print(f"\nNew PDFs in folder (not in database): {len(new_pdfs)}")
for source in sorted(new_pdfs):
    print(f"  - {source}")

# Check for new chunks from existing PDFs
new_chunks = []
for chunk in chunks_with_ids:
    if chunk.metadata["id"] not in existing_ids:
        new_chunks.append(chunk)

print(f"\nTotal chunks from current PDFs: {len(chunks_with_ids)}")
print(f"New chunks to add: {len(new_chunks)}")

if new_chunks:
    print("\nSample new chunk IDs:")
    for chunk in new_chunks[:5]:
        print(f"  - {chunk.metadata['id']}")








