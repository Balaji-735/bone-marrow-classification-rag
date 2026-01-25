"""
Script to fix corrupted Chroma database.
This will delete the corrupted database and recreate it.
"""

import shutil
import sys
from pathlib import Path

# Path to rag-tutorial-v2-main
project_root = Path(__file__).parent.parent
rag_tutorial_path = project_root / "rag-tutorial-v2-main"
chroma_path = rag_tutorial_path / "chroma"

print("=" * 60)
print("Chroma Database Fix Script")
print("=" * 60)

# Check if chroma directory exists
if chroma_path.exists():
    print(f"\nFound Chroma database at: {chroma_path}")
    response = input("Delete and recreate? (y/n): ")
    
    if response.lower() == 'y':
        try:
            # Delete the corrupted database
            print("\nDeleting corrupted database...")
            shutil.rmtree(chroma_path)
            print("✓ Database deleted successfully")
            
            # Now run populate_database.py
            print("\nRepopulating database...")
            print("Please run: cd rag-tutorial-v2-main && python populate_database.py")
            print("\nNote: If populate_database.py still fails, there may be a ChromaDB version issue.")
            print("Try: pip install --upgrade chromadb")
            
        except Exception as e:
            print(f"\n✗ Error deleting database: {e}")
            print("\nYou may need to manually delete the folder:")
            print(f"  {chroma_path}")
    else:
        print("Cancelled.")
else:
    print(f"\nChroma database not found at: {chroma_path}")
    print("You can create it by running:")
    print("  cd rag-tutorial-v2-main && python populate_database.py")

print("\n" + "=" * 60)






