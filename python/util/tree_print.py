#!/usr/bin/env python3
"""
Print a tree view of directory structure.
Usage: python tree_print.py [directory] [max_depth]
"""

import os
import sys
from pathlib import Path

# Handle encoding for TD's Python (cp1252 by default)
# Try to use UTF-8, fall back to ASCII if needed
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f'{size_bytes:.1f}{unit}'
        size_bytes /= 1024
    return f'{size_bytes:.1f}TB'


def print_tree(directory, prefix='', max_depth=None, current_depth=0, show_size=True, use_unicode=True):
    """
    Recursively print directory tree structure.
    
    Args:
        directory: Path to directory to print
        prefix: Prefix for tree formatting
        max_depth: Maximum depth to traverse (None for unlimited)
        current_depth: Current recursion depth
        show_size: Whether to show file sizes
        use_unicode: Whether to use Unicode box-drawing characters
    """
    if max_depth is not None and current_depth >= max_depth:
        return
    
    # Define characters based on unicode support
    if use_unicode:
        branch = '├── '
        last_branch = '└── '
        extension = '│   '
        last_extension = '    '
    else:
        branch = '+-- '
        last_branch = '+-- '
        extension = '|   '
        last_extension = '    '
    
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        print(f'{prefix}{last_branch}[Permission Denied]')
        return
    
    # Separate dirs and files
    dirs = [e for e in entries if os.path.isdir(os.path.join(directory, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(directory, e))]
    
    # Print directories first
    for i, dirname in enumerate(dirs):
        is_last_dir = (i == len(dirs) - 1) and len(files) == 0
        connector = last_branch if is_last_dir else branch
        print(f'{prefix}{connector}{dirname}/')
        
        next_ext = last_extension if is_last_dir else extension
        next_dir = os.path.join(directory, dirname)
        print_tree(next_dir, prefix + next_ext, max_depth, current_depth + 1, show_size, use_unicode)
    
    # Print files
    for i, filename in enumerate(files):
        is_last = i == len(files) - 1
        connector = last_branch if is_last else branch
        filepath = os.path.join(directory, filename)
        
        if show_size:
            try:
                size = os.path.getsize(filepath)
                size_str = f' ({format_size(size)})'
            except OSError:
                size_str = ' (??)'
        else:
            size_str = ''
        
        print(f'{prefix}{connector}{filename}{size_str}')


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else 'tox'
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    directory = Path(directory).resolve()
    
    if not directory.exists():
        print(f'Error: {directory} does not exist')
        sys.exit(1)
    
    if not directory.is_dir():
        print(f'Error: {directory} is not a directory')
        sys.exit(1)
    
    # Try UTF-8 first, fall back to ASCII if encoding issues occur
    use_unicode = True
    try:
        print(f'{directory.name}/')
        print_tree(str(directory), '', max_depth, use_unicode=True)
    except UnicodeEncodeError:
        print(f'\n{directory.name}/')
        print_tree(str(directory), '', max_depth, use_unicode=False)


if __name__ == '__main__':
    main()
