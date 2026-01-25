"""
Script to rename dataset folders to match expected class names.
"""
from pathlib import Path
from src.config import RAW_DATA_DIR, CLASSES

print("Renaming dataset folders...")
print(f"Source directory: {RAW_DATA_DIR}\n")

# Map of possible folder name patterns to class names
folder_mappings = {
    'BLA': ['BLA 0001-1000', 'BLA', 'blast', 'Blast'],
    'EOS': ['EOS 0001-1000', 'EOS', 'eosinophil', 'Eosinophil'],
    'LYT': ['LYT 0001-1000', 'LYT', 'lymphocyte', 'Lymphocyte'],
    'MON': ['MON 0001-1000', 'MON', 'monocyte', 'Monocyte'],
    'NGS': ['NGS 0001-1000', 'NGS', 'neutrophil', 'Neutrophil'],
    'NIF': ['NIF 0001-1000', 'NIF', 'immature', 'Immature'],
    'PMO': ['PMO 0001-1000', 'PMO', 'promyelocyte', 'Promyelocyte']
}

renamed_count = 0

for class_name in CLASSES:
    target_dir = RAW_DATA_DIR / class_name
    
    # Check if already correctly named
    if target_dir.exists():
        print(f"[OK] {class_name}: Already exists")
        continue
    
    # Try to find folder with matching pattern
    found = False
    for pattern in folder_mappings[class_name]:
        source_dir = RAW_DATA_DIR / pattern
        if source_dir.exists():
            print(f"Renaming '{pattern}' -> '{class_name}'")
            source_dir.rename(target_dir)
            print(f"[OK] Successfully renamed to {class_name}")
            renamed_count += 1
            found = True
            break
    
    if not found:
        print(f"[ERROR] {class_name}: No matching folder found")

print(f"\nRenamed {renamed_count} folders.")
print("\nVerifying structure...")

# Verify
all_good = True
for class_name in CLASSES:
    class_dir = RAW_DATA_DIR / class_name
    if class_dir.exists():
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + \
                 list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.bmp"))
        print(f"[OK] {class_name}: {len(images)} images")
    else:
        print(f"[ERROR] {class_name}: Not found!")
        all_good = False

if all_good:
    print("\n[SUCCESS] Dataset structure is correct!")
else:
    print("\n[WARNING] Some folders are missing. Please check manually.")

