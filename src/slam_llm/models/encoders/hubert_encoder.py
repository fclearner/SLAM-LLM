import copy
from typing import List, Optional, Tuple

import torch
from typeguard import typechecked

from slam_llm.models.encoders.abs_encoder import AbsEncoder


class TorchAudioHuBERTPretrainEncoder(AbsEncoder):
    """Torch Audio Hubert encoder module.

    Args:
        extractor_mode: Operation mode of feature extractor.
            Valid values are "group_norm" or "layer_norm".
        extractor_conv_layer_config: Configuration of convolution layers in feature
            extractor. List of convolution configuration,
            i.e. [[output_channel, kernel_size, stride], ...]
        extractor_conv_bias: Whether to include bias term to each convolution
            operation.
        encoder_embed_dim: The dimension of embedding in encoder.
        encoder_projection_dropout: The dropout probability applied after the input
            feature is projected to "encoder_embed_dim".
        encoder_pos_conv_kernel: Kernel size of convolutional positional embeddings.
        encoder_pos_conv_groups: Number of groups of convolutional positional
            embeddings.
        encoder_num_layers: Number of self attention layers in transformer block.
        encoder_num_heads: Number of heads in self attention layers.
        encoder_attention_dropout: Dropout probability applied after softmax in
            self-attention layer.
        encoder_ff_interm_features: Dimension of hidden features in feed forward layer.
        encoder_ff_interm_dropout: Dropout probability applied in feedforward layer.
        encoder_dropout: Dropout probability applied at the end of feed forward layer.
        encoder_layer_norm_first: Control the order of layer norm in transformer layer
            and each encoder layer. If True, in transformer layer, layer norm is
            applied before features are fed to encoder layers.
        encoder_layer_drop: Probability to drop each encoder layer during training.
        mask_prob: Probability for each token to be chosen as start of the span
            to be masked.
        mask_selection: How to choose the mask length.
            Options: [static, uniform, normal, poisson].
        mask_other: Secondary mask argument (used for more complex distributions).
        mask_length: The lengths of the mask.
        no_mask_overlap: Whether to allow masks to overlap.
        mask_min_space: Minimum space between spans (if no overlap is enabled).
        mask_channel_prob: (float): The probability of replacing a feature with 0.
        mask_channel_selection: How to choose the mask length for channel masking.
            Options: [static, uniform, normal, poisson].
        mask_channel_other: Secondary mask argument for channel masking(used for more
            complex distributions).
        mask_channel_length: Minimum space between spans (if no overlap is enabled)
            for channel masking.
        no_mask_channel_overlap: Whether to allow channel masks to overlap.
        mask_channel_min_space: Minimum space between spans for channel
            masking(if no overlap is enabled).
        skip_masked: If True, skip computing losses over masked frames.
        skip_nomask: If True, skip computing losses over unmasked frames.
        num_classes: The number of classes in the labels.
        final_dim: Project final representations and targets to final_dim.
        feature_grad_mult: The factor to scale the convolutional feature extraction
            layer gradients by. The scale factor will not affect the forward pass.
        finetuning: Whether to finetuning the model with ASR or other tasks.
        freeze_encoder_updates: The number of steps to freeze the encoder parameters
            in ASR finetuning.
    Hubert specific Args:
        Please refer to:
        https://pytorch.org/audio/stable/generated/torchaudio.models.hubert_pretrain_model.html#torchaudio.models.hubert_pretrain_model
    """

    @typechecked
    def __init__(
        self,
        input_size: int = None,
        extractor_mode: str = "group_norm",
        extractor_conv_layer_config: Optional[List[List[int]]] = [
            [512, 10, 5],
            [512, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
            [512, 2, 2],
            [512, 2, 2],
        ],
        extractor_conv_bias: bool = False,
        encoder_embed_dim: int = 768,
        encoder_projection_dropout: float = 0.1,
        encoder_pos_conv_kernel: int = 128,
        encoder_pos_conv_groups: int = 16,
        encoder_num_layers: int = 12,
        encoder_num_heads: int = 12,
        encoder_attention_dropout: float = 0.1,
        encoder_ff_interm_features: int = 3072,
        encoder_ff_interm_dropout: float = 0.0,
        encoder_dropout: float = 0.1,
        encoder_layer_norm_first: bool = False,
        encoder_layer_drop: float = 0.05,
        mask_prob: float = 0.8,
        mask_selection: str = "static",
        mask_other: float = 0.0,
        mask_length: int = 10,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        mask_channel_prob: float = 0.0,
        mask_channel_selection: str = "static",
        mask_channel_other: float = 0.0,
        mask_channel_length: int = 10,
        no_mask_channel_overlap: bool = False,
        mask_channel_min_space: int = 1,
        skip_masked: bool = False,
        skip_nomask: bool = False,
        num_classes: int = 100,
        final_dim: int = 256,
        feature_grad_mult: Optional[float] = 0.1,
        finetuning: bool = False,
        freeze_encoder_updates: int = 0,
        training: bool = False,
    ):
        super().__init__()
        try:
            import torchaudio
        except Exception as e:
            print("Error: torchaudio is not properly installed.")
            print("Please install torchaudio")
            raise e

        self._output_size = encoder_embed_dim

        self.hubert_pretrain_model = torchaudio.models.hubert_pretrain_model(
            extractor_mode=extractor_mode,
            extractor_conv_layer_config=extractor_conv_layer_config,
            extractor_conv_bias=extractor_conv_bias,
            encoder_embed_dim=encoder_embed_dim,
            encoder_projection_dropout=encoder_projection_dropout,
            encoder_pos_conv_kernel=encoder_pos_conv_kernel,
            encoder_pos_conv_groups=encoder_pos_conv_groups,
            encoder_num_layers=encoder_num_layers,
            encoder_num_heads=encoder_num_heads,
            encoder_attention_dropout=encoder_attention_dropout,
            encoder_ff_interm_features=encoder_ff_interm_features,
            encoder_ff_interm_dropout=encoder_ff_interm_dropout,
            encoder_dropout=encoder_dropout,
            encoder_layer_norm_first=encoder_layer_norm_first,
            encoder_layer_drop=encoder_layer_drop,
            mask_prob=mask_prob,
            mask_selection=mask_selection,
            mask_other=mask_other,
            mask_length=mask_length,
            no_mask_overlap=no_mask_overlap,
            mask_min_space=mask_min_space,
            mask_channel_prob=mask_channel_prob,
            mask_channel_selection=mask_channel_selection,
            mask_channel_other=mask_channel_other,
            mask_channel_length=mask_channel_length,
            no_mask_channel_overlap=no_mask_channel_overlap,
            mask_channel_min_space=mask_channel_min_space,
            skip_masked=skip_masked,
            skip_nomask=skip_nomask,
            num_classes=num_classes,
            final_dim=final_dim,
            feature_grad_mult=feature_grad_mult,
        )
        self.pretrained_params = copy.deepcopy(self.hubert_pretrain_model.state_dict())

        self.finetuning = finetuning
        self.training = training
        if finetuning:
            for p in self.hubert_pretrain_model.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False
        self.register_buffer("global_step", torch.LongTensor([0]))
        self.freeze_encoder_updates = freeze_encoder_updates

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        ys_pad: torch.Tensor = None,
        ys_pad_length: torch.Tensor = None,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Hubert Pretrain Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """

        if not self.finetuning:
            return self._pretraining_forward(xs_pad, ilens, ys_pad)
        else:
            if self.training:
                return self._finetuning_forward(xs_pad, ilens)
            else:
                return self._eval_forward(xs_pad, ilens)

    def _pretraining_forward(self, xs_pad, ilens, ys_pad):
        assert ys_pad is not None
        (
            logit_m,
            logit_u,
            feature_penalty,
        ) = self.hubert_pretrain_model.forward(xs_pad, ys_pad, ilens)

        return logit_m, logit_u, feature_penalty

    def _finetuning_forward(self, xs_pad, ilens):
        def get_padding_mask(input, lengths):
            """get_padding_mask() from torchaudio.models.wav2vec2.components"""
            batch_size, max_len, _ = input.shape
            mask = (
                torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
                >= lengths[:, None]
            )
            return mask

        # manually add the steps. It is not accurate.
        # TODO(simpleoier): to introduce the global update steps into encoder module
        self.global_step += 1
        if self.global_step <= self.freeze_encoder_updates:
            with torch.no_grad():
                x, out_len = self.hubert_pretrain_model.wav2vec2.feature_extractor(
                    xs_pad, ilens
                )
                padding_mask = get_padding_mask(x, out_len)
                (
                    x,
                    attention_mask,
                ) = self.hubert_pretrain_model.wav2vec2.encoder._preprocess(x, out_len)
                x, _ = self.hubert_pretrain_model.mask_generator(x, padding_mask)
                x = self.hubert_pretrain_model.wav2vec2.encoder.transformer(
                    x, attention_mask=attention_mask
                )
        else:
            with torch.no_grad():
                x, out_len = self.hubert_pretrain_model.wav2vec2.feature_extractor(
                    xs_pad, ilens
                )
                padding_mask = get_padding_mask(x, out_len)

            (
                x,
                attention_mask,
            ) = self.hubert_pretrain_model.wav2vec2.encoder._preprocess(x, out_len)
            x, _ = self.hubert_pretrain_model.mask_generator(x, padding_mask)
            x = self.hubert_pretrain_model.wav2vec2.encoder.transformer(
                x, attention_mask=attention_mask
            )
        return x, (~padding_mask).long().sum(dim=1), None

    def _eval_forward(self, xs_pad, ilens):
        x, lengths = self.hubert_pretrain_model.wav2vec2.feature_extractor(
            xs_pad, ilens
        )
        x = self.hubert_pretrain_model.wav2vec2.encoder(x, lengths)
        return x, lengths, None

    def reload_pretrained_parameters(self):
        self.hubert_pretrain_model.load_state_dict(self.pretrained_params, strict=False)
        logging.info("Pretrained Hubert model parameters reloaded!")
