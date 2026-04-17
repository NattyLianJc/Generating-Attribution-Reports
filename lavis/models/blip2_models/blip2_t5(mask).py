"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration


class MaskPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=14, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj_32 = nn.Linear(num_patches, 32)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2)
        x = self.proj_32(x).transpose(1, 2)
        return x


@registry.register_model("blip2_t5")
class Blip2T5(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xl_vitL: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xl_vitL": "configs/models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            t5_model="google/flan-t5-xl",
            prompt="",
            max_txt_len=150,  # 32
            apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer()  # bert-base 在Blip2Base里

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        # 在Blip2Base里

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
                # param.data = param.data.bfloat16()
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # 把加进去的conv解冻
        for name, param in self.visual_encoder.named_parameters():
            if "conv_mask" in name:
                param.requires_grad = True

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # 加一个mask的patch embed
        self.mask_patch_embed = MaskPatchEmbed(img_size=img_size)

        # 冻Q
        # for name, param in self.Qformer.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.Qformer.bert.encoder.named_parameters():
        #     if int(name.split(".")[1]) > 5:
        #         param.requires_grad = True

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            # param.requires_grad = False
            # param.data = param.data.bfloat16()
            if "lm_head" in name:
                param.requires_grad = True
            elif "decoder" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def forward(self, samples):
        # image = samples["image"]
        image = samples["image"][:, :-1, :, :]
        mask = samples["image"][:, -1, :, :]
        mask = mask.unsqueeze(dim=1)

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))  # [b, 257, 1408]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )  # [b, 257]

        mask_embeds = self.mask_patch_embed(mask)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # [b, 32, 768]

        query_tokens_with_mask = torch.cat((query_tokens, mask_embeds), dim=1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens_with_mask,  # query_tokens
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])  # [b, 32, 2048]
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)  # [b, 32]

        text_input = [""] * image.size(0)
        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                text_input,  # samples["text_input"]\text_input\samples["mask_input"]
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens = self.t5_tokenizer(
                samples["caption"],  # samples["text_output"]\samples["caption"]\samples["label"]
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)  # [b, 33]

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)  # [b, 1, 2048]
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)  #

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            # mlm token
            '''
            input_tokens_mlm = self.t5_tokenizer(
                samples["mask_input"],  # samples["text_input"]\text_input\samples["mask_input"]
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            output_tokens_mlm = self.t5_tokenizer(
                samples["label"],  # samples["text_output"]\samples["caption"]\samples["label"]
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts_mlm = torch.cat([atts_t5, input_tokens_mlm.attention_mask], dim=1)  # [b, 33]

            targets_mlm = output_tokens_mlm.input_ids.masked_fill(
                output_tokens_mlm.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            idxs = (output_tokens_mlm.input_ids >= 32000) & (output_tokens_mlm.input_ids <= 32099)
            targets_mlm = targets_mlm.masked_fill(idxs, -100)

            inputs_embeds_mlm = self.t5_model.encoder.embed_tokens(input_tokens_mlm.input_ids)  # [b, 1, 2048]
            inputs_embeds_mlm = torch.cat([inputs_t5, inputs_embeds_mlm], dim=1)  #

            outputs_mlm = self.t5_model(
                inputs_embeds=inputs_embeds_mlm,
                attention_mask=encoder_atts_mlm,
                decoder_attention_mask=output_tokens_mlm.attention_mask,
                return_dict=True,
                labels=targets_mlm,
            )
            loss = outputs.loss + outputs_mlm.loss
            '''

        return {"loss": loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=30,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # image = samples["image"]

        image = samples["image"][:, :-1, :, :]
        mask = samples["image"][:, -1, :, :]
        mask = mask.unsqueeze(dim=1)

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        mask_embeds = self.mask_patch_embed(mask)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # [b, 32, 768]

        query_tokens_with_mask = torch.cat((query_tokens, mask_embeds), dim=1)

        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens_with_mask,  # query_tokens
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        else:
            assert len(prompt) == image.size(
                0
            ), "The number of prompts must be equal to the batch size."

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    def predict_answers(
            self,
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=128,
            answer_list=None,
            prompt="",
            length_penalty=-1,
            **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        # if isinstance(samples["text_input"], str):
        #     samples["text_input"] = [samples["text_input"]]
        # if prompt:
        #     text_input = [prompt.format(question) for question in samples["text_input"]]
        # else:
        #     text_input = samples["text_input"]
        text_input = [""] * image.size(0)

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 60)  # 32

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
