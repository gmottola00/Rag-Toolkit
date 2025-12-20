#!/usr/bin/env python3
"""
Fix all incorrect imports in the rag-toolkit codebase.

This script addresses multiple import issues:
1. Remove 'core' prefix from rag imports: rag_toolkit.core.rag -> rag_toolkit.rag
2. Remove 'infra' nesting: rag_toolkit.infra.embedding.ollama -> rag_toolkit.infra.embedding.ollama
3. Fix any remaining src. imports
"""

import argparse
import re
from pathlib import Path

# Mapping of incorrect -> correct imports
IMPORT_FIXES = [
    # Fix: rag_toolkit.core.rag -> rag_toolkit.rag
    (r'from rag_toolkit\.core\.rag', 'from rag_toolkit.rag'),
    (r'import rag_toolkit\.core\.rag', 'import rag_toolkit.rag'),
    
    # Fix: rag_toolkit.core.index -> rag_toolkit.core.index (keep this)
    # Fix: rag_toolkit.core.chunking -> rag_toolkit.core.chunking (keep this)
    
    # Fix any remaining src. imports
    (r'from src\.', 'from rag_toolkit.'),
    (r'import src\.', 'import rag_toolkit.'),
]


def fix_imports_in_file(filepath: Path, root_path: Path, dry_run: bool = False) -> int:
    """Fix imports in a single file. Returns number of changes."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        changes = 0
        
        for pattern, replacement in IMPORT_FIXES:
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                if not dry_run:
                    try:
                        rel_path = filepath.relative_to(root_path)
                    except ValueError:
                        rel_path = filepath.name
                    print(f"  {rel_path}: {pattern} -> {replacement} ({count} changes)")
                content = new_content
                changes += count
        
        if changes > 0 and not dry_run:
            filepath.write_text(content, encoding='utf-8')
        
        return changes
    
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Fix all imports in rag-toolkit")
    parser.add_argument(
        "--root",
        type=str,
        default="src/rag_toolkit",
        help="Root directory to scan (default: src/rag_toolkit)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    
    args = parser.parse_args()
    root_path = Path(args.root)
    
    if not root_path.exists():
        print(f"❌ Directory not found: {root_path}")
        return 1
    
    print(f"🔍 Scanning {root_path} for Python files...")
    py_files = list(root_path.rglob("*.py"))
    print(f"📁 Found {len(py_files)} Python files\n")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")
    else:
        print("✏️  Fixing imports...\n")
    
    total_changes = 0
    files_modified = 0
    
    for filepath in py_files:
        changes = fix_imports_in_file(filepath, root_path, dry_run=args.dry_run)
        if changes > 0:
            files_modified += 1
            total_changes += changes
    
    print("\n" + "=" * 60)
    if args.dry_run:
        print("🔍 DRY RUN SUMMARY:")
    else:
        print("✅ SUMMARY:")
    print(f"   Files processed: {len(py_files)}")
    print(f"   Files modified:  {files_modified}")
    print(f"   Total changes:   {total_changes}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
