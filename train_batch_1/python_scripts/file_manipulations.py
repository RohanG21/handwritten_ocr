import shutil
import os 
from itertools import islice
from pathlib import Path

def clear_folder(folder):
    folder_val = Path(folder)
    folder_val.mkdir(parents=True,exist_ok=True)
    
    for item in folder.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)   # deletes the subfolder and everything inside it
        else:
            item.unlink()         # deletes a file (or a symlink, without following it)


def copy_first_n_percent_files(source_dir, dest_dir, n):
    source = Path(source_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Only regular files (skip subfolders), sorted by name
    files = sorted(p for p in source.iterdir() if p.is_file())

    for file in files[:n]:
        shutil.copy2(file, dest / file.name)  # copy2 also keeps timestamps

def copy_n_file_lines(source,dest,n):
    source_file = Path(source)
    dest_file = Path(dest)

    with open(source_file, "r", encoding="utf-8") as src, \
         open(dest_file, "w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if i >= n:
                break
            dst.write(line)


def delete_n_file_lines(file,n):
    if not isinstance(file,Path):
        return

    with open(source_file, "w", encoding="utf-8") as src:
        for line in islice(src, n, None):   # skip the first n lines
            line = line.strip()             # remove leading/trailing spaces, tabs, \r, \n
            if line:                        # drop blank lines
                src.write(line + "\n")      # exactly one newline per record

    

    



