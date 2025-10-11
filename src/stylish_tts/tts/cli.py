import sys
import click
from scipy.io.wavfile import write
import numpy as np
from .stylish_model import StylishModel
import pyloudnorm as pyln


@click.group("stylish-tts")
def cli():
    pass


@cli.command(
    "speak",
    short_help="Use a Stylish TTS model to convert text from stdin to audio, one utterance per line.",
)
@click.argument("model", type=str)
@click.argument("out", type=str)
@click.option(
    "--lang",
    type=str,
    default="phonemes",
    help="ISO 639 language code to use for G2P or 'phonemes' for no G2P",
)
def speak_document(model, out, lang):
    if lang != "phonemes":
        exit("Only phoneme input supported for now")

    model = StylishModel(model)
    meter = pyln.Meter(model.sample_rate())  # create BS.1770 meter
    results = []
    for line in sys.stdin.readlines():
        tokens = model.tokenize(line.strip())
        audio = model.generate_speech(tokens)
        loudness = meter.integrated_loudness(audio)
        audio = pyln.normalize.loudness(audio, loudness, -20.0)
        audio = np.multiply(audio, 32768).astype(np.int16)
        results.append(audio)
        sys.stderr.write(".")
        sys.stderr.flush()

    sys.stderr.write("\n")
    sys.stderr.flush()
    combined = np.concatenate(results)
    print("Saving to:", out)
    write(out, model.sample_rate(), combined)
