#!/usr/bin/env python3
"""
Fix import paths in the codebase.

This script automatically converts all 'src.' imports to 'rag_toolkit.' imports
throughout the codebase, making it packageable as a proper Python library.

Usage:
    python scripts/fix_imports.py
    
    # Dry run (see changes without applying)
    python scripts/fix_imports.py --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator


def find_python_files(root_dir: Path) -> Iterator[Path]:
    """Find all Python files in the directory tree."""
    return root_dir.rglob("*.py")


def fix_imports_in_file(file_path: Path, dry_run: bool = False) -> tuple[int, list[str]]:
    """
    Fix imports in a single file.
    
    Returns:
        Tuple of (num_changes, list_of_changes)
    """
    content = file_path.read_text(encoding="utf-8")
    original_content = content
    changes = []
    
    # Pattern 1: from src.module import ...
    pattern1 = r"from src\."
    replacement1 = "from rag_toolkit."
    new_content, count1 = re.subn(pattern1, replacement1, content)
    if count1 > 0:
        changes.append(f"  - Changed 'from src.' → 'from rag_toolkit.' ({count1} times)")
    content = new_content
    
    # Pattern 2: import src.module
    pattern2 = r"import src\."
    replacement2 = "import rag_toolkit."
    new_content, count2 = re.subn(pattern2, replacement2, content)
    if count2 > 0:
        changes.append(f"  - Changed 'import src.' → 'import rag_toolkit.' ({count2} times)")
    content = new_content
    
    total_changes = count1 + count2
    
    if total_changes > 0 and not dry_run:
        file_path.write_text(content, encoding="utf-8")
    
    return total_changes, changes


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix import paths in the codebase")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without applying them"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("src/rag_toolkit"),
        help="Root directory to process (default: src/rag_toolkit)"
    )
    args = parser.parse_args()
    
    root_dir = args.root
    if not root_dir.exists():
        print(f"❌ Error: Directory {root_dir} does not exist")
        return
    
    print(f"🔍 Scanning {root_dir} for Python files...")
    python_files = list(find_python_files(root_dir))
    print(f"📁 Found {len(python_files)} Python files")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified\n")
    else:
        print("\n✏️  Fixing imports...\n")
    
    total_files_changed = 0
    total_changes = 0
    
    for file_path in python_files:
        num_changes, changes = fix_imports_in_file(file_path, dry_run=args.dry_run)
        
        if num_changes > 0:
            total_files_changed += 1
            total_changes += num_changes
            
            rel_path = file_path.relative_to(root_dir.parent)
            print(f"📝 {rel_path}")
            for change in changes:
                print(change)
            print()
    
    # Summary
    print("=" * 60)
    if args.dry_run:
        print(f"🔍 DRY RUN SUMMARY:")
    else:
        print(f"✅ SUMMARY:")
    print(f"   Files processed: {len(python_files)}")
    print(f"   Files modified:  {total_files_changed}")
    print(f"   Total changes:   {total_changes}")
    print("=" * 60)
    
    if args.dry_run and total_changes > 0:
        print("\nTo apply changes, run without --dry-run flag")


if __name__ == "__main__":
    main()
