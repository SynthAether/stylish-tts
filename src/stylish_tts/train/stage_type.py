from typing import Callable, List, Optional, Tuple
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import torchaudio
from einops import rearrange
from stylish_tts.train.loss_log import LossLog, build_loss_log
from stylish_tts.train.utils import (
    print_gpu_vram,
    log_norm,
    length_to_mask,
    leaky_clamp,
)
from typing import List
from stylish_tts.train.losses import multi_phase_loss

stages = {}


def is_valid_stage(name):
    return name in stages


def valid_stage_list():
    return list(stages.keys())


class StageType:
    def __init__(
        self,
        next_stage: Optional[str],
        train_fn: Callable,
        validate_fn: Callable,
        train_models: List[str],
        eval_models: List[str],
        discriminators: List[str],
        inputs: List[str],
    ):
        self.next_stage: Optional[str] = next_stage
        self.train_fn: Callable = train_fn
        self.validate_fn: Callable = validate_fn
        self.train_models: List[str] = train_models
        self.eval_models: List[str] = eval_models
        self.discriminators = discriminators
        self.inputs: List[str] = inputs


def make_list(tensor: torch.Tensor) -> List[torch.Tensor]:
    result = []
    for i in range(tensor.shape[0]):
        result.append(tensor[i])
    return result


class AcousticStep:
    def __init__(
        self,
        batch,
        train,
        loss_log,
        *,
        use_predicted_pe,
        use_textual_style,
        predict_audio,
    ):
        self.batch = batch
        self.train = train
        self.log = loss_log
        with torch.no_grad():
            self.mel, _ = calculate_mel(
                batch.audio_gt,
                train.to_mel,
                train.normalization.mel_log_mean,
                train.normalization.mel_log_std,
            )
            self.energy = log_norm(
                self.mel.unsqueeze(1),
                train.normalization.mel_log_mean,
                train.normalization.mel_log_std,
            ).squeeze(1)
            self.pitch = normalize_pitch(
                batch.pitch, train.f0_log2_mean, train.f0_log2_std
            )
            self.voiced = (batch.pitch > 10).float()
        self.pe_text_encoding, _, _ = train.model.pe_text_encoder(
            batch.text, batch.text_length
        )
        if use_textual_style:
            self.pe_style = train.model.pe_text_style_encoder(
                self.pe_text_encoding, batch.text_length
            )
        else:
            self.pe_style = train.model.pe_mel_style_encoder(self.mel.unsqueeze(1))
        self.pred_pitch, self.pred_energy, self.pred_voiced = (
            train.model.pitch_energy_predictor(
                self.pe_text_encoding,
                batch.text_length,
                batch.alignment,
                self.pe_style,
            )
        )
        if predict_audio:
            if use_predicted_pe:
                self.pred = train.model.speech_predictor(
                    batch.text,
                    batch.text_length,
                    batch.alignment,
                    self.pred_pitch,
                    self.pred_energy,
                    self.pred_voiced,
                )
            else:
                self.pred = train.model.speech_predictor(
                    batch.text,
                    batch.text_length,
                    batch.alignment,
                    self.pitch,
                    self.energy,
                    self.voiced,
                )
            (
                self.target_spec,
                self.pred_spec,
                self.target_phase,
                self.pred_phase,
                self.target_fft,
                self.pred_fft,
            ) = train.multi_spectrogram(
                target=batch.audio_gt, pred=self.pred.audio.squeeze(1)
            )
        else:
            self.pred = None
            self.target_spec = None
            self.pred_spec = None
            self.target_phase = None
            self.pred_phase = None
            self.target_fft = None
            self.pred_fft = None

    def mel_loss(self):
        self.train.stft_loss(
            target_list=self.target_spec, pred_list=self.pred_spec, log=self.log
        )

    def multi_phase_loss(self):
        self.log.add_loss(
            "multi_phase",
            multi_phase_loss(
                self.pred_phase, self.target_phase, self.train.model_config.n_fft
            ),
        )

    def generator_loss(self, disc_index):
        self.log.add_loss(
            "generator",
            self.train.generator_loss(
                target_list=self.target_fft,
                pred_list=self.pred_fft,
                used=["mrd"],
                index=disc_index,
            ).mean(),
        )

    def slm_loss(self):
        self.log.add_loss(
            "slm",
            self.train.wavlm_loss(self.batch.audio_gt.detach(), self.pred.audio),
        )

    def magphase_loss(self):
        self.train.magphase_loss(self.pred, self.batch.audio_gt, self.log)

    def pitch_loss(self):
        self.log.add_loss(
            "pitch",
            torch.nn.functional.smooth_l1_loss(self.pitch, self.pred_pitch),
        )

    def voiced_loss(self):
        self.log.add_loss(
            "voiced",
            torch.nn.functional.binary_cross_entropy(self.pred_voiced, self.voiced),
        )

    def energy_loss(self):
        self.log.add_loss(
            "energy",
            torch.nn.functional.smooth_l1_loss(self.energy, self.pred_energy),
        )


##### Alignment #####


def train_alignment(
    batch, model, train, probing, disc_index
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    log = build_loss_log(train)
    mel, mel_length = calculate_mel(
        batch.audio_gt,
        train.to_align_mel,
        train.normalization.mel_log_mean,
        train.normalization.mel_log_std,
    )
    mel = rearrange(mel, "b f t -> b t f")
    ctc, _ = model.text_aligner(mel, mel_length)
    train.stage.optimizer.zero_grad()
    loss_ctc = train.align_loss(
        ctc, batch.text, mel_length, batch.text_length, step_type="train"
    )

    log.add_loss(
        "align_loss",
        loss_ctc,
    )
    train.accelerator.backward(log.backwards_loss())
    return log.detach(), None, None


@torch.no_grad()
def validate_alignment(batch, train):
    log = build_loss_log(train)
    mel, mel_length = calculate_mel(
        batch.audio_gt,
        train.to_align_mel,
        train.normalization.mel_log_mean,
        train.normalization.mel_log_std,
    )
    mel = rearrange(mel, "b f t -> b t f")
    ctc, _ = train.model.text_aligner(mel, mel_length)
    train.stage.optimizer.zero_grad()

    loss_ctc = train.align_loss(
        ctc, batch.text, mel_length, batch.text_length, step_type="eval"
    )

    blank = train.model_config.text_encoder.tokens
    logprobs = rearrange(ctc, "t b k -> b t k")
    confidence_total = 0.0
    confidence_count = 0
    for i in range(mel.shape[0]):
        _, scores = torchaudio.functional.forced_align(
            log_probs=logprobs[i].unsqueeze(0).contiguous(),
            targets=batch.text[i, : batch.text_length[i].item()].unsqueeze(0),
            input_lengths=mel_length[i].unsqueeze(0),
            target_lengths=batch.text_length[i].unsqueeze(0),
            blank=blank,
        )
        confidence_total += scores.exp().sum()
        confidence_count += scores.shape[-1]
    log.add_loss("confidence", confidence_total / confidence_count)
    log.add_loss("align_loss", loss_ctc)
    return log, None, None, None


stages["alignment"] = StageType(
    next_stage=None,
    train_fn=train_alignment,
    validate_fn=validate_alignment,
    train_models=["text_aligner"],
    eval_models=[],
    discriminators=[],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
    ],
)

##### Acoustic #####


def train_acoustic(
    batch, model, train, probing, disc_index
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    """Train a single batch for the acoustic stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=False,
        use_textual_style=False,
        predict_audio=True,
    )
    train.stage.optimizer.zero_grad()

    step.mel_loss()
    step.multi_phase_loss()
    step.generator_loss(disc_index)
    step.slm_loss()
    step.magphase_loss()
    step.pitch_loss()
    step.voiced_loss()
    step.energy_loss()

    train.accelerator.backward(log.backwards_loss())
    return log.detach(), detach_all(step.target_fft), detach_all(step.pred_fft)


@torch.no_grad()
def validate_acoustic(batch, train):
    """Validate a single batch for the acoustic stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=False,
        use_textual_style=False,
        predict_audio=True,
    )

    step.mel_loss()
    step.multi_phase_loss()
    step.pitch_loss()
    step.voiced_loss()
    step.energy_loss()

    return log, batch.alignment[0], make_list(step.pred.audio), batch.audio_gt


stages["acoustic"] = StageType(
    next_stage="textual",
    train_fn=train_acoustic,
    validate_fn=validate_acoustic,
    train_models=[
        "speech_predictor",
        "pitch_energy_predictor",
        "pe_text_encoder",
        "pe_mel_style_encoder",
    ],
    eval_models=[],
    # discriminators=[],
    discriminators=["mrd0", "mrd1", "mrd2"],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)

##### Textual #####


def train_textual(
    batch, model, train, probing, disc_index
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    """Train a single batch for the textual stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=True,
        use_textual_style=False,
        predict_audio=True,
    )
    train.stage.optimizer.zero_grad()

    step.mel_loss()
    step.multi_phase_loss()
    step.generator_loss(disc_index)
    step.magphase_loss()
    step.pitch_loss()
    step.voiced_loss()
    step.energy_loss()

    train.accelerator.backward(log.backwards_loss())
    return log.detach(), detach_all(step.target_fft), detach_all(step.pred_fft)


@torch.no_grad()
def validate_textual(batch, train):
    """Validate a single batch for the textual stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=True,
        use_textual_style=False,
        predict_audio=True,
    )
    train.stage.optimizer.zero_grad()

    step.mel_loss()
    step.multi_phase_loss()
    step.pitch_loss()
    step.voiced_loss()
    step.energy_loss()

    return log, batch.alignment[0], make_list(step.pred.audio), batch.audio_gt


stages["textual"] = StageType(
    next_stage="style",
    train_fn=train_textual,
    validate_fn=validate_textual,
    train_models=[
        "pitch_energy_predictor",
        "pe_text_encoder",
        "pe_mel_style_encoder",
    ],
    eval_models=["speech_predictor"],
    # discriminators=[],
    discriminators=["mrd0", "mrd1", "mrd2"],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)

##### Style #####


def train_style(
    batch, model, train, probing, disc_index
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    """Train a single batch for the style stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=True,
        use_textual_style=True,
        predict_audio=False,
    )
    pe_mel_style = train.model.pe_mel_style_encoder(step.mel.unsqueeze(1))
    train.stage.optimizer.zero_grad()

    log.add_loss(
        "style",
        torch.nn.functional.smooth_l1_loss(step.pe_style, pe_mel_style) * 10,
    )
    step.pitch_loss()
    step.voiced_loss()
    step.energy_loss()

    train.accelerator.backward(log.backwards_loss())
    return log.detach(), None, None


@torch.no_grad()
def validate_style(batch, train):
    """Validate a single batch for the style stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=True,
        use_textual_style=True,
        predict_audio=True,
    )
    pe_mel_style = train.model.pe_mel_style_encoder(step.mel.unsqueeze(1))
    train.stage.optimizer.zero_grad()

    step.mel_loss()
    log.add_loss(
        "style", torch.nn.functional.smooth_l1_loss(step.pe_style, pe_mel_style) * 10
    )
    step.pitch_loss()
    step.voiced_loss()
    step.energy_loss()

    return log, batch.alignment[0], make_list(step.pred.audio), batch.audio_gt


stages["style"] = StageType(
    next_stage="duration",
    train_fn=train_style,
    validate_fn=validate_style,
    train_models=[
        "pe_text_style_encoder",
    ],
    eval_models=[
        "pe_mel_style_encoder",
        "pitch_energy_predictor",
        "pe_text_encoder",
        "speech_predictor",
    ],
    discriminators=[],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)


##### Duration #####


def train_duration(
    batch, model, train, probing, disc_index
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    targets = train.duration_processor.align_to_class(batch.alignment)
    duration = model.duration_predictor(batch.text, batch.text_length)
    target_dur = batch.alignment.sum(dim=-1)

    train.stage.optimizer.zero_grad()
    duration_loss = 0
    for i in range(duration.shape[0]):
        dur = train.duration_processor.prediction_to_duration(
            duration[i], batch.text_length[i]
        )
        dur = dur[: batch.text_length[i]]
        duration_loss += F.smooth_l1_loss(dur, target_dur[i, : batch.text_length[i]])

    loss_ce, loss_cdw = train.duration_loss(duration, targets, batch.text_length)

    log = build_loss_log(train)
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", duration_loss / duration.shape[0])  # loss_cdw)
    train.accelerator.backward(log.backwards_loss())

    return log.detach(), None, None


@torch.no_grad()
def validate_duration(batch, train):
    pe_text_encoding, _, _ = train.model.pe_text_encoder(batch.text, batch.text_length)
    pe_text_style = train.model.pe_text_style_encoder(
        pe_text_encoding, batch.text_length
    )
    duration = train.model.duration_predictor(batch.text, batch.text_length)
    target_dur = batch.alignment.sum(dim=-1)
    results = []
    duration_loss = 0
    for i in range(duration.shape[0]):
        dur = train.duration_processor.prediction_to_duration(
            duration[i], batch.text_length[i]
        )
        dur = dur[: batch.text_length[i]]
        duration_loss += F.smooth_l1_loss(dur, target_dur[i, : batch.text_length[i]])
        alignment = train.duration_processor.duration_to_alignment(dur)
        alignment = rearrange(alignment, "t a -> 1 t a")
        pred_pitch, pred_energy, pred_voiced = train.model.pitch_energy_predictor(
            pe_text_encoding[i : i + 1, :, : batch.text_length[i]],
            batch.text_length[i : i + 1],
            alignment,
            pe_text_style[i : i + 1],
        )
        pred = train.model.speech_predictor(
            batch.text[i : i + 1, : batch.text_length[i]],
            batch.text_length[i : i + 1],
            alignment,
            pred_pitch,
            pred_energy,
            pred_voiced,
        )
        audio = rearrange(pred.audio, "1 1 l -> l")
        results.append(audio)
    log = build_loss_log(train)
    loss_ce, loss_cdw = train.duration_loss(
        duration,
        train.duration_processor.align_to_class(batch.alignment),
        batch.text_length,
    )
    log.add_loss("duration_ce", loss_ce)
    log.add_loss("duration", duration_loss / duration.shape[0])  # loss_cdw)
    # log.add_loss("duration", loss_dur)

    return log.detach(), alignment[0], results, batch.audio_gt


stages["duration"] = StageType(
    next_stage="joint",
    train_fn=train_duration,
    validate_fn=validate_duration,
    train_models=[
        "duration_predictor",
    ],
    eval_models=[
        "pitch_energy_predictor",
        "speech_predictor",
        "pe_text_encoder",
        "pe_text_style_encoder",
    ],
    discriminators=[],
    inputs=[
        "text",
        "text_length",
        "alignment",
        "audio_gt",
    ],
)


##### Joint #####


def train_joint(
    batch, model, train, probing, disc_index
) -> Tuple[LossLog, Optional[torch.Tensor]]:
    """Train a single batch for the joint stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=True,
        use_textual_style=True,
        predict_audio=True,
    )
    pe_mel_style = train.model.pe_mel_style_encoder(step.mel.unsqueeze(1))
    train.stage.optimizer.zero_grad()

    step.mel_loss()
    step.multi_phase_loss()
    step.generator_loss(disc_index)
    step.slm_loss()
    step.magphase_loss()
    log.add_loss(
        "style",
        torch.nn.functional.smooth_l1_loss(step.pe_style, pe_mel_style) * 10,
    )
    step.pitch_loss()
    step.voiced_loss()
    step.energy_loss()

    train.accelerator.backward(log.backwards_loss())
    return (
        log.detach(),
        detach_all(step.target_fft),
        detach_all(step.pred_fft),
    )


@torch.no_grad()
def validate_joint(batch, train):
    """Validate a single batch for the joint stage"""
    log = build_loss_log(train)
    step = AcousticStep(
        batch,
        train,
        log,
        use_predicted_pe=True,
        use_textual_style=True,
        predict_audio=True,
    )
    pe_mel_style = train.model.pe_mel_style_encoder(step.mel.unsqueeze(1))
    train.stage.optimizer.zero_grad()

    step.mel_loss()
    step.multi_phase_loss()
    step.pitch_loss()
    step.voiced_loss()
    step.energy_loss()

    return log, batch.alignment[0], make_list(step.pred.audio), batch.audio_gt


stages["joint"] = StageType(
    next_stage=None,
    train_fn=train_joint,
    validate_fn=validate_joint,
    train_models=[
        "pe_text_style_encoder",
        "pitch_energy_predictor",
        "pe_text_encoder",
        "speech_predictor",
    ],
    eval_models=["pe_mel_style_encoder"],
    discriminators=["mrd0", "mrd1", "mrd2"],
    # discriminators=[],
    inputs=[
        "text",
        "text_length",
        "audio_gt",
        "pitch",
        "alignment",
    ],
)


#########################


def detach_all(spec_list):
    result = []
    for item in spec_list:
        result.append(item.detach())
    return result


@torch.no_grad()
def calculate_mel(audio, to_mel, mean, std):
    mel = to_mel(audio)
    mel = (torch.log(1e-5 + mel) - mean) / std
    # STFT returns audio_len // hop_len + 1, so we strip off the extra here
    mel = mel[:, :, : (mel.shape[-1] - mel.shape[-1] % 2)]
    mel_length = torch.full(
        [audio.shape[0]], mel.shape[2], dtype=torch.long, device=audio.device
    )
    return mel, mel_length


def normalize_pitch(f0, log_f0_mean, log_f0_std):
    """
    Normalizes f0 using pre-calculated log-scale z-score statistics.
    """

    voiced = f0 > 10

    # Use torch or numpy log2 based on input type
    log_f0 = torch.log2(f0 + 1e-8)

    # Standardize using the calculated stats
    normed_f0 = (log_f0 - log_f0_mean) / log_f0_std

    # Set unvoiced parts to 0 (which now represents the mean of the normed space)
    normed_f0 = normed_f0 * voiced
    return normed_f0


def denormalize_pitch(
    normed_f0,
    log_f0_mean,
    log_f0_std,
    min_hz=30,
    max_hz=600,
):
    """
    Denormalizes f0 from z-score + log-scale, WITH a safety clamp.
    """
    # De-standardize
    log_f0 = normed_f0 * log_f0_std + log_f0_mean

    # Convert back from log-scale
    f0 = 2**log_f0
    voiced = f0 > 10
    f0 = leaky_clamp(f0, min_f=min_hz, max_f=max_hz, slope=0.01)

    # Set unvoiced parts to 0
    f0 = f0 * voiced

    return f0
