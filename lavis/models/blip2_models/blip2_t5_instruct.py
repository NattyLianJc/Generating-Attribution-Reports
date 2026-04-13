"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import string
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast, BertTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutput

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel

from lavis.models.mask_decoder import MaskDecoder
from lavis.models.TwoWayTransformer import TwoWayTransformer
from vit.models.modeling import VisionTransformer, CONFIGS


class TagGatingMechanism(nn.Module):
    def __init__(self, input_dim):
        super(TagGatingMechanism, self).__init__()
        self.gate = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tag_embeddings):
        gate_values = self.sigmoid(self.gate(tag_embeddings))
        return gate_values


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss


@registry.register_model("blip2_t5_instruct")
class Blip2T5Instruct(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - flant5xl
        - flant5xxl
        - flant5base
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2/blip2_instruct_flant5xl.yaml",
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl.yaml",
        "flant5base": "configs/models/blip2/blip2_instruct_flant5base.yaml",
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
            max_txt_len=128,
            max_output_txt_len=256,
            apply_lemmatizer=False,
            num_few_shot_examples=0,
            few_shot_prob=0,
            qformer_text_input=True,
            num_classes=21,
            multi_label_classifier_path="",
            bert_folder="",
            num_hidden_layers=12
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        for name, param in self.visual_encoder.named_parameters():
            if "Adapter" in name:
                param.requires_grad = True

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 
            self.visual_encoder.num_features, 
            bert_folder=bert_folder,
            num_hidden_layers=num_hidden_layers 
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.seg_proj = nn.Linear(
            self.Qformer.config.hidden_size, 224 * 224
        )
        self.seg_reshape = nn.Unflatten(dim=1, unflattened_size=(224, 224))

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_folder)
        self.bert_model = BertModel.from_pretrained(bert_folder)

        self.qformer_text_input = qformer_text_input

        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=1408,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=1408,
        )

        self.multi_label = ['eye', 'pupil', 'lip', 'mouth', 'nose', 'ear', 'cheek', 'forehead', 'chin', 'jaw', 'eyebrow', 'tooth', 'tongue', 'beard', 'mustache', 'skin', 'hair', 'wrinkle', 'freckle', 'scar', 'dimple']

        if multi_label_classifier_path:
            self.multi_label_classifier = self.load_vit_pretrained_weights(multi_label_classifier_path, num_classes=num_classes)
            for param in self.multi_label_classifier.parameters():
                param.requires_grad = False

        self.cross_attention_layer = nn.MultiheadAttention(embed_dim=1408, num_heads=8)
        self.gate_cross_attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=8)

        self.prompt_proj = nn.Linear(self.t5_model.config.hidden_size, 1408)
        self.fc_layer = nn.Linear(768, 1)
        self.trans_layer = nn.Linear(1408, 768)
        self.mlm_head = nn.Linear(768, self.tokenizer.vocab_size)
        self.text_proj = nn.Linear(self.t5_model.config.hidden_size, 1408)

        self.training_stage = "finetune"

    def load_vit_pretrained_weights(self, weights_path, num_classes=21):
        state_dict = torch.load(weights_path, map_location="cpu")
        multi_label_classifier = VisionTransformer(CONFIGS["ViT-H_14"], img_size=224, num_classes=num_classes)
        multi_label_classifier.load_state_dict(state_dict)            
        return multi_label_classifier

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2, bert_folder="", num_hidden_layers=12):
        encoder_config = BertConfig.from_pretrained(bert_folder)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        Qformer = BertLMHeadModel(config=encoder_config) 
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        return Qformer, query_tokens

    def get_prompt_embeddings(self, prompts):
        text_input = self.t5_tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=21,
            return_tensors="pt",
        ).to(self.device)
        prompt_embeds = self.t5_model.encoder.embed_tokens(text_input.input_ids)
        return prompt_embeds

    def get_predicted_tag_embeddings(self, predicted_labels, threshold, logits):
        tag_embeddings_list = []
        batch_size = predicted_labels.shape[0]

        for i in range(batch_size):
            present_labels = [self.multi_label[j] for j in range(len(self.multi_label)) if predicted_labels[i, j] > threshold]
            if present_labels:
                tag_inputs = self.bert_tokenizer(
                    present_labels,
                    padding='longest',
                    truncation=True,
                    return_tensors="pt"
                ).to(logits.device)

                tag_embedding = self.bert_model(**tag_inputs).last_hidden_state
                tag_embeddings_list.append(tag_embedding)
            else:
                tag_embeddings_list.append(None)

        return tag_embeddings_list

    def _get_multimodal_context(self, image, gt_labels=None):
        """
        Extracted internal method to handle the shared logic of extracting image embeddings, 
        processing tag gating, calculating gate loss, and formatting prompts.
        """
        bs = image.size(0)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

        with torch.no_grad():
            logits = self.multi_label_classifier(image)[0]

        threshold = 0.5
        predicted_labels = torch.sigmoid(logits)
        tag_embeddings_list = self.get_predicted_tag_embeddings(predicted_labels, threshold, logits)

        gate_values_list = []
        for i in range(bs):
            if tag_embeddings_list[i] is not None:
                tag_embeddings = tag_embeddings_list[i]
                
                query_embeds = tag_embeddings.permute(1, 0, 2)
                key_value_embeds = image_embeds[i, 1:, :].unsqueeze(1)
                key_value_embeds = key_value_embeds.expand(-1, query_embeds.size(1), -1)
                key_value_embeds = self.trans_layer(key_value_embeds)

                cross_attn_output, _ = self.gate_cross_attention_layer(query_embeds, key_value_embeds, key_value_embeds)
                cross_attn_output = cross_attn_output.permute(1, 0, 2)

                gate_values = torch.sigmoid(self.fc_layer(cross_attn_output))
                gate_values = gate_values.mean(dim=1)
                gate_values_list.append(gate_values)
            else:
                gate_values_list.append(None)

        gated_labels = predicted_labels.clone()
        prompts_filtered_labels = []

        gate_loss = 0.0
        if gt_labels is not None:
            for i in range(bs):
                if gate_values_list[i] is not None:
                    positive_indices = torch.where(predicted_labels[i] > threshold)[0]
                    gt_present_labels = gt_labels[i, positive_indices]
                    gate_loss += F.binary_cross_entropy_with_logits(
                        gate_values_list[i].squeeze(-1), 
                        gt_present_labels,
                        reduction='mean'
                    )
            gate_loss /= bs

        for i in range(bs):
            if gate_values_list[i] is not None:
                mask = (predicted_labels[i] > threshold).float()
                gate_values = gate_values_list[i].squeeze(-1)

                gated_label_update = gated_labels[i].clone()
                positive_indices = torch.where(mask > 0)[0]

                gated_label_update[positive_indices] *= gate_values
                gated_labels[i] = gated_label_update

                filtered_labels = [self.multi_label[j] for j in positive_indices]
            else:
                filtered_labels = []
            prompts_filtered_labels.append(filtered_labels)

        prompts = []
        labels = [] 
        for filtered_labels in prompts_filtered_labels:
            if filtered_labels:
                prompt = f"The following facial areas may have been modified by AI: {', '.join(filtered_labels)}. These areas might have been manipulated either by replacing the entire face or by altering specific facial features. Please identify these areas and describe why they appear unnatural."
                label = ' '.join(filtered_labels)
            else:
                prompt = "The following facial areas may have been modified by AI. These areas might have been manipulated either by replacing the entire face or by altering specific facial features. Please identify these areas and describe why they appear unnatural."
                label = ''
            prompts.append(prompt)
            labels.append(label)

        if gt_labels is not None:
            return image_embeds, prompts, labels, gate_loss
        return image_embeds, prompts, labels

    def _get_mask_prediction(self, image_embeds, labels):
        """
        Extracted internal method to handle prompt projection and mask decoding.
        """
        prompt_embeds = self.get_prompt_embeddings(labels)
        prompt_embeds = self.prompt_proj(prompt_embeds.float()) 

        cls_token = image_embeds[:, 0:1, :]
        image_embeds_no_cls = image_embeds[:, 1:, :].permute(1, 0, 2)
        prompt_embeds = prompt_embeds.permute(1, 0, 2)
        
        cross_attn_output, _ = self.cross_attention_layer(prompt_embeds, image_embeds_no_cls, image_embeds_no_cls)
        cross_attn_output = cross_attn_output.permute(1, 0, 2)

        cross_attn_output = torch.cat((cls_token, cross_attn_output), dim=1)

        seg_out = self.mask_decoder(cross_attn_output)
        seg_out = torch.sigmoid(seg_out)
        
        return seg_out

    def forward(self, samples):
        if self.training_stage == "pretrain":
            return self.forward_pretrain(samples)
        elif self.training_stage == "finetune":
            return self.forward_finetune(samples)
        else:
            raise ValueError(f"Unknown training stage: {self.training_stage}")

    def create_mlm_labels(self, input_ids, max_seq_len):
        region_terms = {
            "eye": ["eye", "eyes", "ocular", "optic"],
            "pupil": ["pupil", "pupils"],
            "lip": ["lip", "lips", "labium", "labia"],
            "mouth": ["mouth", "oral cavity", "buccal"],
            "nose": ["nose", "nasal", "nostril", "nostrils"],
            "ear": ["ear", "ears", "auricle", "pinna"],
            "cheek": ["cheek", "cheeks", "buccal", "malar"],
            "forehead": ["forehead", "frontal", "brow"],
            "chin": ["chin", "mentum"],
            "jaw": ["jaw", "mandible", "maxilla"],
            "eyebrow": ["eyebrow", "eyebrows"],
            "tooth": ["tooth", "teeth", "dentition"],
            "tongue": ["tongue", "glossa", "lingua"],
            "beard": ["beard", "facial hair"],
            "mustache": ["mustache", "moustache"],
            "skin": ["skin", "derma", "epidermis"],
            "hair": ["hair", "locks", "tresses"],
            "wrinkle": ["wrinkle", "wrinkles", "line", "lines"],
            "freckle": ["freckle", "freckles", "lentigo", "lentigines"],
            "scar": ["scar", "scars", "cicatrix", "cicatrices"],
            "dimple": ["dimple", "dimples"]
        }

        batch_size, original_seq_len = input_ids.shape
        mlm_labels = torch.full((batch_size, max_seq_len), fill_value=-100, dtype=torch.long, device=input_ids.device)

        for i in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            for j, token in enumerate(tokens):
                for key, synonyms in region_terms.items():
                    if token in synonyms:
                        mlm_labels[i, j] = input_ids[i, j]
                        input_ids[i, j] = self.tokenizer.mask_token_id
                        break

        if mlm_labels.shape[1] < max_seq_len:
            pad = torch.full((batch_size, max_seq_len - mlm_labels.shape[1]), fill_value=-100, dtype=torch.long, device=mlm_labels.device)
            mlm_labels = torch.cat([mlm_labels, pad], dim=1)
        elif mlm_labels.shape[1] > max_seq_len:
            mlm_labels = mlm_labels[:, :max_seq_len]

        return input_ids, mlm_labels

    def forward_pretrain(self, samples):
        image = samples["image"]
        text_input = samples["caption"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_features = image_embeds[:, 0, :]

        text_tokenized = self.tokenizer(
            text_input,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        max_seq_len = text_tokenized.input_ids.shape[1]
        masked_input_ids, mlm_labels = self.create_mlm_labels(text_tokenized.input_ids.clone(), max_seq_len)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_tokenized.attention_mask], dim=1)

        query_output = self.Qformer.bert(
            masked_input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device),
            return_dict=True,
        )

        logits = self.mlm_head(query_output.last_hidden_state[:, query_tokens.size(1):, :])
        mlm_loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            mlm_labels.view(-1),
            ignore_index=-100
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
        text_features = self.text_proj(inputs_t5.mean(dim=1))

        contrastive_loss_fn = ContrastiveLoss(temperature=0.07)
        contrastive_loss = contrastive_loss_fn(image_features, text_features)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            output_tokens = self.t5_output_tokenizer(
                samples["caption"],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            vocab_size = self.t5_tokenizer.vocab_size
            targets = torch.where(
                (targets >= 0) & (targets < vocab_size),
                targets,
                torch.full_like(targets, -100)
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            lm_loss = outputs.loss

            if torch.isnan(outputs.loss):
                logging.warning("NaN detected in T5 outputs.loss!")

            gt_labels = samples["multilabel"]
            labels = []
            for gt_label in gt_labels:
                gt_label_tags = [self.multi_label[i] for i in range(len(gt_label)) if gt_label[i] == 1]
                label = ' '.join(gt_label_tags)
                labels.append(label)

            seg_out = self._get_mask_prediction(image_embeds, labels)
            mask_label = samples["mask"].squeeze(dim=1)

            pos_weight = torch.ones([1]).to(image.device) * 2
            criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            seg_loss = criterion_G(seg_out, mask_label)

        total_loss = 2 * mlm_loss + 1 * lm_loss + 1 * seg_loss + 1 * contrastive_loss
        return {"loss": total_loss, "mlm_loss": mlm_loss, "lm_loss": lm_loss, "seg_loss": seg_loss, "contrastive_loss": contrastive_loss}

    def forward_finetune(self, samples):
        image = samples["image"]
        gt_labels = samples.get("multilabel", None)
        
        # 1. DRY applied: Fetch multimodal context
        image_embeds, prompts, labels, gate_loss = self._get_multimodal_context(image, gt_labels)

        # 2. DRY applied: Calculate mask predictions and segmentation loss
        seg_out = self._get_mask_prediction(image_embeds, labels)
        mask_label = samples["mask"].squeeze(dim=1)
        pos_weight = torch.ones([1]).to(image.device) * 2
        criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        seg_loss = criterion_G(seg_out, mask_label)

        # 3. Proceed with Qformer and T5 logic
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompts,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        fs_embeds, fs_atts = None, None
        if self.few_shot_prob > 0 and "few_shot_samples" in samples.keys():
            fs_embeds, fs_atts = self.prepare_few_shot_embeds(samples['few_shot_samples'])

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                prompts, 
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            
            output_tokens = self.t5_output_tokenizer(
                samples["caption"],
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            if fs_embeds is not None:
                inputs_embeds = torch.cat([fs_embeds, inputs_embeds], dim=1)
                encoder_atts = torch.cat([fs_atts, encoder_atts], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            lm_loss = outputs.loss
            t5_logits = outputs.logits 

            loss = lm_loss + seg_loss + 0.5 * gate_loss

        return {
            "loss": loss, 
            "lm_loss": lm_loss, 
            "seg_loss": seg_loss, 
            "seg_out": seg_out,
            "qformer_features": inputs_t5, 
            "t5_logits": t5_logits   
        }

    def prepare_few_shot_embeds(self, samples):
        this_n_fs = random.choices(
            list(range(self.num_few_shot_examples + 1)),
            weights=[1 - self.few_shot_prob] + [self.few_shot_prob / self.num_few_shot_examples] * self.num_few_shot_examples
        )[0]

        if this_n_fs == 0:
            return None, None

        images = []
        text_input = []
        for sample in samples:
            for n in range(this_n_fs):
                images.append(sample['image'][n])
                text_input.append(sample['text_input'][n])
        images = torch.stack(images, dim=0)
        image = images

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                text_input,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        if this_n_fs > 1:
            encoder_atts = encoder_atts.reshape(encoder_atts.size(0) // this_n_fs, encoder_atts.size(1) * this_n_fs)
            inputs_embeds = inputs_embeds.reshape(inputs_embeds.size(0) // this_n_fs, inputs_embeds.size(1) * this_n_fs, inputs_embeds.size(2))

        return inputs_embeds, encoder_atts

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
            return_seg=False
    ):
        image = samples["image"]
        bs = image.size(0)

        # 1. DRY applied: Fetch multimodal context
        image_embeds, prompts, labels = self._get_multimodal_context(image)
        
        seg_out = None
        if return_seg:
            # 2. DRY applied: Fetch mask prediction if requested
            seg_out = self._get_mask_prediction(image_embeds, labels)

        # 3. Proceed with standard generation logic
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompts,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_t5, atts_t5 = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids, attention_mask=Qformer_atts, query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds, encoder_attention_mask=frame_atts, return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens, encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts, return_dict=True,
                    )

                frame_inputs_t5 = self.t5_proj(frame_query_output.last_hidden_state[:, :query_tokens.size(1), :])
                frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
        else:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids, attention_mask=Qformer_atts, query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens, encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts, return_dict=True,
                )

            inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            prompts, padding="longest", truncation=True, max_length=self.max_txt_len, return_tensors="pt"
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
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if return_seg:
            return output_text, seg_out
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
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                        for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
            return_seg=False
        )

        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
            self,
            samples,
            candidates,
            n_segments=1,
    ):
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
            self,
            samples,
            candidates,
            n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_t5, atts_t5 = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_t5 = self.t5_proj(frame_query_output.last_hidden_state[:, :query_tokens.size(1), :])
                frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)
        output_tokens = self.t5_tokenizer(
            candidates, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        n_cands = len(candidates)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
            )

            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                this_encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0].clone(),
                )

                this_encoder_outputs['last_hidden_state'] = this_encoder_outputs[0].repeat_interleave(seg_len, dim=0)
                this_encoder_atts = encoder_atts.repeat_interleave(seg_len, dim=0)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                this_output_tokens_ids = output_tokens.input_ids[start_i:end_i].repeat(bs, 1)
                this_output_tokens_atts = output_tokens.attention_mask[start_i:end_i].repeat(bs, 1)

                this_targets = this_output_tokens_ids.masked_fill(this_output_tokens_ids == self.t5_tokenizer.pad_token_id, -100)

                outputs = self.t5_model(
                    encoder_outputs=this_encoder_outputs,
                    attention_mask=this_encoder_atts,
                    decoder_attention_mask=this_output_tokens_atts,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )
                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

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
        # 1. Basic model settings
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size", 224)
        num_query_token = cfg.get("num_query_token", 32)
        t5_model = cfg.get("t5_model")

        # 2. Precision and freezing settings
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        # 3. Text and generation settings
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 60)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        # 4. Training strategy settings
        num_few_shot_examples = cfg.get("num_few_shot_examples", 0)
        few_shot_prob = cfg.get("few_shot_prob", 0.0)
        qformer_text_input = cfg.get("qformer_text_input", True)
        num_hidden_layers = cfg.get("num_hidden_layers", 12)

        # 5. Path and custom module settings (Crucial fix here)
        # These must be retrieved from cfg and passed to __init__
        bert_folder = cfg.get("bert_folder")
        multi_label_classifier_path = cfg.get("multi_label_classifier_path")
        num_classes = cfg.get("num_classes", 21)

        # Initialize the model with ALL necessary parameters
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
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_prob=few_shot_prob,
            qformer_text_input=qformer_text_input,
            num_hidden_layers=num_hidden_layers,
            bert_folder=bert_folder,
            multi_label_classifier_path=multi_label_classifier_path,
            num_classes=num_classes
        )

        # Load weights from the path specified in 'pretrained' in yaml
        model.load_checkpoint_from_config(cfg)

        return model