import pandas as pd

with open('data/raw/hypothyroid.data', 'r') as f:
    for i, line in enumerate(f):
        print(f"Line {i+1}: {repr(line)}")
        if i >= 4:
            break
        