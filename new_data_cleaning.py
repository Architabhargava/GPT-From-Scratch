import pandas as pd
import re
import os

file_path = "dataset/premchand.tsv"
df = pd.read_csv(file_path, sep="\t")

# Drop NaNs and strip whitespace
df['Title of Work'] = df['Title of Work'].fillna('').astype(str).str.strip()
df['Text'] = df['Text'].fillna('').astype(str).str.strip()

# Combine title and text with proper formatting
# Title on its own line, then story, then 2 line breaks
df['combined'] = df['Title of Work'] + "\n" + df['Text'] + "\n\n"

# Concatenate everything
all_text = "\n".join(df['combined'].tolist())

# Optional: Normalize weird Unicode spaces, multiple newlines, etc.
all_text = re.sub(r'\r', '', all_text)
all_text = re.sub(r'\n{3,}', '\n\n', all_text)  # collapse 3+ newlines to 2
all_text = re.sub(r'[ \t]+', ' ', all_text)     # collapse multiple spaces
all_text = all_text.strip()

# Ensure output directory exists
output_path = "dataset/new_input.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(all_text)

print(f" Cleaned dataset saved to {output_path}")
print(f"Total characters: {len(all_text):,}")
print(f"Total lines: {len(all_text.splitlines()):,}")
