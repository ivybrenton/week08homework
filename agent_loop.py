#!/usr/bin/env python3
"""
Programmatic prompt -> generate -> test -> feedback loop using GeminiSimpleAPI.

"""

import subprocess
import sys
import shutil
import time
from pathlib import Path

# Allow running from this directory without installing the package.
_FILES_DIR = Path(__file__).resolve().parent.parent / "files"
if str(_FILES_DIR) not in sys.path:
    sys.path.insert(0, str(_FILES_DIR))

from gemini_simple_api import GeminiSimpleAPI  # noqa: E402

TASK_DIR = Path(__file__).parent
TEST_DIR = TASK_DIR / "tests"
TEST_FILE = TEST_DIR / "test_bayes_factor.py"
SOURCE_FILE = TASK_DIR / "bayes_factor.py"
PROMPT_FILE = TASK_DIR / "task.txt"

# Set test_bayes_factor.py to read-only(!)
# Note this won't do anything if the agent can run as root.
TEST_FILE.chmod(0o444)

# Modifiable parameters

MODEL = "gemma-4-31b-it"
MAX_ATTEMPTS = 10
INCLUDE_TEST_FILE = False


def run_tests() -> tuple[int, str]:
    result = subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", TEST_DIR],
        cwd=TASK_DIR,
        capture_output=True,
        text=True,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


client = GeminiSimpleAPI(
    api_key_file=Path("/workspace/secrets/gemini.json"),
    model=MODEL,
    working_dir=TASK_DIR,
    protected_directories=[TEST_DIR],
)

prompt_text = PROMPT_FILE.read_text()

prompt_text += (
    "\n\nOnly write or update bayes_factor.py.\n"
    "Do not modify the test file.\n"
    "Do not modify anything inside the tests directory.\n"
    "Do not include markdown fences.\n"
    "Return only valid Python code."
)

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\n=== Attempt {attempt} ===")

    try:
        files, notes = client.prompt(
            prompt=prompt_text,
            attachments=[TEST_FILE] if INCLUDE_TEST_FILE else [],
            verbose=True,
        )
    except Exception as e:
        print(f"Model call failed: {e}")
        print("Waiting before trying again.")
        time.sleep(5)
        continue

    # Here you could re-insert the test file if it was modified.

    code, output = run_tests()
    print(f"Output: {output}")

    # Archive the attempt
    (TASK_DIR / f"attempt_{attempt}").mkdir(parents=True, exist_ok=True)
    (TASK_DIR / f"attempt_{attempt}" / "output.txt").write_text(output)
    (TASK_DIR / f"attempt_{attempt}" / "prompt.txt").write_text(prompt_text)
    for file in files:
        shutil.copy(file, TASK_DIR / f"attempt_{attempt}" / file.name)

    # input("Press Enter to continue...")
    if code == 0:
        print(f"\nTests passed on attempt {attempt}.")
        break

    prompt_text += (
        f"\n\n## Attempt {attempt} failed\n"
        f"```\n{output}\n```"
        "\nFix the failures above. Only modify bayes_factor.py."
    )

    time.sleep(3)

else:
    print(f"\nStopped after {MAX_ATTEMPTS} attempts; tests still failing.")
    sys.exit(1)