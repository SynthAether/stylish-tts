from stylish_tts.train.cli import cli as train_cli_function
from stylish_tts.tts.cli import cli as tts_cli_function


def train_cli() -> None:
    train_cli_function()


def tts_cli() -> None:
    tts_cli_function()
