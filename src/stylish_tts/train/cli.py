import os.path as osp
import click
import importlib.resources
import multiprocessing
import torch

from stylish_tts.lib.config_loader import load_config_yaml, load_model_config_yaml
import stylish_tts.train.config as config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config(config_path):
    if osp.exists(config_path):
        config = load_config_yaml(config_path)
    else:
        # TODO: we may be able to pull it out of the model if a model is passed in instead
        logger.error(f"Config file not found at {config_path}")
        exit(1)
    return config


def get_model_config(model_config_path):
    if len(model_config_path) == 0:
        path = importlib.resources.files(config) / "model.yml"
        f_model = path.open("r", encoding="utf-8")
    else:
        if osp.exists(model_config_path):
            f_model = open(model_config_path, "r", encoding="utf-8")
        else:
            logger.error(f"Config file not found at {model_config_path}")
            exit(1)
    result = load_model_config_yaml(f_model)
    f_model.close()
    return result


##############################################################################


@click.group("stylish-train")
def cli():
    """Prepare a dataset, train a model, or convert a model to ONNX:

    In order to train, first you `train-align` to create an alignment model, `align` to use that model to generate alignments, `pitch` to generate pitch estimation for the dataset. At this point, as long as your dataset does not change, you do not need to re-run any of these stages again.

    Once you have pre-cached alignments and pitches, you can `train` your model, and finally `convert` your model to ONNX for inference.

    """
    # multiprocessing.set_start_method("forkserver")
    # torch.multiprocessing.set_sharing_strategy("file_system")
    # print("Setting multiprocessing start method to spawn.")


##### train-align #####


@cli.command(
    "train-align",
    short_help="Train an alignment model to use for pre-caching alignments.",
)
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option("--out", type=str, help="Output directory for logs and checkpoints")
@click.option(
    "--checkpoint",
    default="",
    type=str,
    help="Path to a model checkpoint to load before training.",
)
@click.option(
    "--reset-stage",
    "reset_stage",
    is_flag=True,
    help="If loading a checkpoint, do not skip epochs and data.",
)
def train_align(config_path, model_config_path, out, checkpoint, reset_stage):
    """Train alignment model

    <config_path> is your main configuration file and the resulting alignment model will be stored at <path>/<alignment_model_path> as specified in the dataset section.
    """
    print("Train alignment...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)
    from stylish_tts.train.train import train_model

    train_model(
        config,
        model_config,
        out,
        "alignment",
        checkpoint,
        reset_stage,
        config_path,
        model_config_path,
    )


##### align #####


@cli.command(
    short_help="Use a pretrained alignment model to create a cache of alignments for training."
)
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
def align(config_path, model_config_path):
    """Align dataset

    <config_path> is your main configuration file. Use an alignment model to precache the alignments for your dataset. <config_path> is your main configuration file and the alignment model will be loaded from <path>/<alignment_model_path>. The alignments are saved to <path>/<alignment_path> as specified in the dataset section. 'scores_val.txt' and 'scores_train.txt' containing confidence scores for each segment will be written to the dataset <path>.
    """
    print("Calculate alignment...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)
    from stylish_tts.train.dataprep.align_text import align_text

    align_text(config, model_config)


##### pitch #####


@cli.command(short_help="Create a cache of pitches to use for training.")
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "-k",
    "--workers",
    default=8,
    type=int,
    help="Number of worker threads to use for calculation",
)
@click.option(
    "--method",
    default="pyworld",
    type=str,
    help="Method used to calculate. 'pyworld' (CPU based, traditional), 'rmvpe' (GPU based, ML model)",
)
def pitch(config_path, model_config_path, workers, method):
    """Calculate pitch for a dataset

    <config_path> is your main configuration file. Calculates the fundamental frequencies for every segment in your dataset. The pitches are saved to the <path>/<pitch_path> from the dataset section of the config file.
    """
    if method != "pyworld" and method != "rmvpe":
        exit("Pitch calculation must either be pyworld or rmvpe")
    print("Calculate pitch...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)
    from stylish_tts.train.dataprep.pitch_extractor import calculate_pitch

    calculate_pitch(config, model_config, method, workers)


##### train #####


@cli.command(short_help="Train a model using the specified configuration.")
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "--out", type=str, help="Output directory for logs, checkpoints, and models"
)
@click.option(
    "--stage",
    default="acoustic",
    type=str,
    help="Training stage should be one of 'acoustic', 'textual', 'style', 'duration'.",
)
@click.option(
    "--checkpoint",
    default="",
    type=str,
    help="Path to a model checkpoint to load before training.",
)
@click.option(
    "--reset-stage",
    "reset_stage",
    is_flag=True,
    help="If loading a checkpoint, do not skip epochs and data.",
)
def train(config_path, model_config_path, out, stage, checkpoint, reset_stage):
    """Train a model

    <config_path> is your main configuration file. Train a Stylish TTS model. You must have already precached alignment and pitch information for the dataset. Stage should be 'acoustic' to begin with unless you are loading a checkpoint.
    """
    print("Train model...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)
    from stylish_tts.train.train import train_model

    train_model(
        config,
        model_config,
        out,
        stage,
        checkpoint,
        reset_stage,
        config_path,
        model_config_path,
    )


##### convert #####


@cli.command(short_help="Convert a model to ONNX for use in inference.")
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "--duration", required=True, type=str, help="Path to write duration model"
)
@click.option("--speech", required=True, type=str, help="Path to write speech model")
@click.option(
    "--checkpoint",
    required=True,
    type=str,
    help="Path to a model checkpoint to load for conversion",
)
def convert(config_path, model_config_path, duration, speech, checkpoint):
    """Convert a model to ONNX

    The converted model will be saved in <out-file>.
    """
    print("Convert to ONNX...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)

    from stylish_tts.train.train_context import Manifest
    from stylish_tts.train.convert_to_onnx import convert_to_onnx
    from stylish_tts.train.models.models import build_model
    from stylish_tts.train.utils import DurationProcessor
    from accelerate import Accelerator
    from accelerate import DistributedDataParallelKwargs
    from stylish_tts.train.losses import DiscriminatorLoss

    duration_processor = DurationProcessor(
        class_count=model_config.duration_predictor.duration_classes,
        max_dur=model_config.duration_predictor.max_duration,
    ).to(config.training.device)
    manifest = Manifest()

    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False, find_unused_parameters=True
    )
    accelerator = Accelerator(
        project_dir=".",
        split_batches=True,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=config.training.mixed_precision,
        step_scheduler_with_optimizer=False,
    )
    model = build_model(model_config)
    for key in model:
        model[key] = accelerator.prepare(model[key])
        model[key].to(config.training.device)

    disc_loss = DiscriminatorLoss(mrd0=model.mrd0, mrd1=model.mrd1, mrd2=model.mrd2)

    from stylish_tts.train.train_context import NormalizationStats

    norm = NormalizationStats()
    accelerator.register_for_checkpointing(config)
    accelerator.register_for_checkpointing(model_config)
    accelerator.register_for_checkpointing(manifest)
    accelerator.register_for_checkpointing(disc_loss)
    accelerator.register_for_checkpointing(norm)

    accelerator.load_state(checkpoint)

    convert_to_onnx(
        model_config,
        duration,
        speech,
        model,
        config.training.device,
        duration_processor,
    )
    # Embed normalization stats into ONNX metadata (only if present in checkpoint)
    from stylish_tts.train.convert_to_onnx import add_meta_data_onnx

    if norm.frames > 0:
        add_meta_data_onnx(speech, "mel_log_mean", str(norm.mel_log_mean))
        add_meta_data_onnx(speech, "mel_log_std", str(norm.mel_log_std))
        logger.info(
            f"Embedded normalization stats from checkpoint: mean={norm.mel_log_mean:.4f}, std={norm.mel_log_std:.4f}"
        )
    else:
        logger.warning(
            "Checkpoint did not contain normalization stats; skipping embedding in ONNX."
        )
