import os
import subprocess
from pathlib import Path

engines_dir = Path("engines")
# We'll analyze them one by one to avoid huge context issues and get better focus.
results = []

for engine_path in engines_dir.glob("*/engine.py"):
    name = engine_path.parent.name
    print(f"Analyzing {name}...")
    code = engine_path.read_text(encoding='utf-8')
    prompt = f"You are a senior software architect. Analyze this Python recommendation engine code: {name}/engine.py. \n" \
             "Identify its core logic, strengths, weaknesses, and any specific refactoring ideas.\n" \
             "RESPOND IN ENGLISH ONLY.\n\n" \
             f"```python\n{code}\n```"
    
    try:
        process = subprocess.Popen(
            ['ollama', 'run', 'gemma4:31b'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        stdout, stderr = process.communicate(input=prompt, timeout=300)
        results.append(f"# Analysis of {name}\n\n{stdout}\n\n")
    except Exception as e:
        results.append(f"# Analysis of {name}\n\nFailed to run Ollama: {e}\n\n")

with open("data/architecture_review.md", "w", encoding='utf-8') as f:
    f.writelines(results)
