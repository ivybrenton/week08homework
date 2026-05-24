#!/usr/bin/env python3

import subprocess
import sys
import shutil
import time
from pathlib import Path

# Allow importing gemini_simple_api.py from ../files
_FILES_DIR = Path(__file__).resolve().parent.parent / "files"
if str(_FILES_DIR) not in sys.path:
    sys.path.insert(0, str(_FILES_DIR))

from gemini_simple_api import GeminiSimpleAPI  # noqa: E402


MODEL = "gemma-4-31b-it"
MAX_ATTEMPTS = 10

TASK_DIR = Path(__file__).parent
TEST_DIR = TASK_DIR / "tests"
TEST_FILE = TEST_DIR / "test_bayes_factor.py"
SOURCE_FILE = TASK_DIR / "bayes_factor.py"
PROMPT_FILE = TASK_DIR / "task.txt"

# Make the test file read-only
TEST_FILE.chmod(0o444)


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

prompt_text += """

Only write or update bayes_factor.py.
Do not modify the test file.
Do not modify anything inside the tests directory.
Do not include markdown fences.
Return only valid Python code.
"""

original_test_file = TEST_FILE.read_text()

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\n=== Attempt {attempt} ===")

    try:
        files, notes = client.prompt(
            prompt=prompt_text,
            attachments=[],
            verbose=True,
        )
    except Exception as e:
        print(f"Model call failed: {e}")
        print("Waiting before trying again.")
        time.sleep(5)
        continue

    if TEST_FILE.read_text() != original_test_file:
        raise RuntimeError("The test file changed. Stopping the loop.")

    code, output = run_tests()
    print(f"Output:\n{output}")

    # Save attempt outputs
    attempt_dir = TASK_DIR / f"attempt_{attempt}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    (attempt_dir / "output.txt").write_text(output)
    (attempt_dir / "prompt.txt").write_text(prompt_text)

    for file in files:
        shutil.copy(file, attempt_dir / file.name)

    if code == 0:
        print(f"\nTests passed on attempt {attempt}.")
        break

    prompt_text += (
        f"\n\n## Attempt {attempt} failed\n"
        f"{output}\n"
        "Fix the failures above. Only modify bayes_factor.py."
    )

    time.sleep(3)

else:
    print(f"\nStopped after {MAX_ATTEMPTS} attempts; tests still failing.")
    sys.exit(1)