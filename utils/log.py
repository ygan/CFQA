"""Simple file logging helpers used by the public LLM wrapper."""

from datetime import datetime
import logging
from pathlib import Path

LOG_DIR = Path("log")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / f"{datetime.now().date()}.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def readable_log(message: str) -> None:
    readable_path = LOG_DIR / f"{datetime.now().date()}.readable"
    with readable_path.open("a", encoding="utf-8") as file:
        file.write("========================\n" * 3)
        file.write(message)
        file.write("\n")
        file.write("========================\n" * 3)
        file.write("\n")


def llm_log(input_message, output_message, **kwargs) -> None:
    content = "\t\tLLM input:\n"
    if isinstance(input_message, list):
        for item in input_message:
            content += f"{item['role']}:\n{item['content']}\n\n"
    else:
        content += str(input_message)
    content += f"\n\t\tLLM output:\n{output_message}"
    if "model" in kwargs:
        content += f"\n\n\t\tLLM Model:\t{kwargs['model']}"
    if "usage" in kwargs:
        content += f"\n\n\t\tUsage(Prompt, Completion, Total Tokens):\t{kwargs['usage']}"
    logging.info(content)
    readable_log(content)
