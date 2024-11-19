from transformers import M2M100ForConditionalGeneration, M2M100Config, Seq2SeqTrainer, DataCollatorForSeq2Seq, GenerationConfig, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutput
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder, shift_tokens_right
from torch import cat, float16
import torch
import utils
from typing import Optional, Tuple, Union
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#class MorphM2M100(torch.nn.Module):
class MorphM2M100(M2M100ForConditionalGeneration):
    #in use
    def __init__(
        self,
        checkpoint,
        morph_encoder_layers=2,
        morph_d_model=512,
        morph_dropout=0.2,
        #morph_encoder_config,
        freeze_base_encoder=True,
        encoder_scheme="embed_dim",
        **kwargs
        ):
        """
        possible encoder_scheme values:
            "embed_dim": concatenate encoder inputs along embedding dimension
        """
        #create base encoder-decoder model and lm_head from pretrained checkpoint
        
        #this is a hack I'm trying to get from_pretrained overload to work
        if checkpoint=='load_empty':
            dummy_cfg = M2M100Config.from_pretrained('facebook/nllb-200-distilled-600m')
            super().__init__(dummy_cfg, **kwargs)
            self.model = None
            self.lm_head = None
            self.morph_encoder = None
            self.projection_layer = None

        else:
            source_model = M2M100ForConditionalGeneration.from_pretrained(checkpoint)
            config_dict = source_model.config.to_dict()
            if 'max_length' in config_dict:
                del config_dict['max_length']
            config = PretrainedConfig.from_dict(config_dict)
            super().__init__(config, **kwargs)
            self.generation_config.max_length=200
            self.model = source_model.model
            self.lm_head = source_model.lm_head
            del source_model
            
            #create new encoder for morph tags from config
            morph_encoder_config = M2M100Config(
            vocab_size=1024,
            encoder_layers=morph_encoder_layers,
            d_model=morph_d_model,
            dropout=morph_dropout,
            encoder_layerdrop=0,
            pad_token_id=0,
            max_position_embeddings=1024,
            scale_embedding=True,
        )
            self.morph_encoder = M2M100Encoder(morph_encoder_config)

            #create projection layer for correcting concatenated dimension
            if encoder_scheme == "embed_dim":
                self.projection_layer = torch.nn.Linear(
                    config.d_model+morph_encoder_config.d_model,
                    config.d_model
                    )

            #freeze parameters in base model encoder
            if freeze_base_encoder:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False


    #for testing
    def base_model_unmodified_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    #in use
    def modified_inner_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        tags: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tags_attention_mask: Optional[torch.Tensor] = None,        
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        #print('start inner forward')
        


        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            #print(f"encoder outputs type: {type(encoder_outputs)}")
            morph_encoder_outputs = self.morph_encoder(
                input_ids=tags,
                attention_mask=tags_attention_mask,
                head_mask=head_mask, # do i need this for the morph encoder? do I need a separate one?
            )
            combined_encoder_outputs = cat((encoder_outputs.last_hidden_state, morph_encoder_outputs.last_hidden_state), dim=-1)
            projected_encoder_outputs = self.projection_layer(combined_encoder_outputs)
            #assert encoder_outputs.last_hidden_state.shape == morph_encoder_outputs.last_hidden_state.shape == projected_encoder_outputs.shape
            encoder_outputs = BaseModelOutput(
                last_hidden_state=projected_encoder_outputs,
                #hidden_states=projected_encoder_outputs[1] if len(projected_encoder_outputs) > 1 else None,
                #attentions=projected_encoder_outputs[2] if len(projected_encoder_outputs) > 2 else None,
                hidden_states=None,
                attentions=None
            )
            #print(f"projected_encoder_outputs shape: {projected_encoder_outputs.shape}")  # Expecting (batch_size, sequence_length, d_model)
            #print(f"attention_mask shape: {attention_mask.shape}")  # Expecting (batch_size, sequence_length)
            #print(f"encoder_outputs shape: {encoder_outputs.last_hidden_state.shape}")  # Expecting (batch_size, sequence_length, d_model)
            #print(f"encoder_outputs[0] shape: {encoder_outputs[0].shape}")  # Expecting (batch_size, sequence_length, d_model)
            #print(f"encoder_outputs last hidden state shape: {encoder_outputs.last_hidden_state.shape}")  # Expecting (batch_size, sequence_length, d_model)

            #print(f"modified encoder outputs type: {type(encoder_outputs)}")


        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        #print('before decoder')
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        #print('after decoder')
        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    #in use
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        tags: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tags_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        #print(f"input ids: {input_ids}")
        #print(f"labels: {labels}")
        #print(f"tags: {tags}")
        #print(f"attention_mask: {attention_mask}")
        #print(f"tags_attention_mask: {tags_attention_mask}")
        #copied from the forward fct from M2M100ForConditionalGeneration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.modified_inner_forward(
        #outputs = self.base_model_unmodified_forward(
            input_ids,
            tags=tags,
            attention_mask=attention_mask,
            tags_attention_mask=tags_attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            # move labels to the correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        #print('after lm head calc')
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )




    @classmethod
    def from_pretrained(
        cls, #what is this
        model_path,
        *model_args,
        **kwargs
    ):
        """
        overload to load second encoder and projection layer along with model
        """
        
        model = MorphM2M100('load_empty')
        
        base_model_path = os.path.join(model_path, "base_model/")

        base_model = M2M100ForConditionalGeneration.from_pretrained(base_model_path, *model_args, **kwargs)
        model.model = base_model.model
        model.lm_head = base_model.lm_head
        model.config = base_model.config
        model.generation_config = base_model.generation_config
        #model = super(MorphM2M100, cls).from_pretrained(base_model_path, *model_args, **kwargs, safetensors=True)
        #model = super().from_pretrained(model_path, *model_args, **kwargs, safetensors=True)
        morph_encoder_path = os.path.join(model_path, "morph_encoder/")
        projection_layer_path = os.path.join(model_path, "projection_layer/projection_layer.pt")
        if os.path.exists(morph_encoder_path):
            model.morph_encoder = M2M100Encoder.from_pretrained(morph_encoder_path, *model_args, **kwargs)
            #model.morph_encoder.load_state_dict(torch.load(morph_encoder_path))

        if os.path.exists(projection_layer_path):
            model.projection_layer = torch.nn.Linear(
                    model.config.d_model+model.morph_encoder.config.d_model,
                    model.config.d_model
                    )
            model.projection_layer.load_state_dict(torch.load(projection_layer_path))

        return model



    
    def save_pretrained(
        self,
        save_directory,
        **kwargs
    ):
        """
        overload to save second encoder and projection layer along with model
        """

        #SAVE THE BASE MODEL
        base_model_path = os.path.join(save_directory, "base_model/")
        #wait I can't do self.model.save_pretrained because it won't include the lm head oof, I have to do super().save
        super().save_pretrained(base_model_path, **kwargs)
        #that should save everything the base model is expecting to see

        #SAVE THE MORPH ENCODER
        morph_encoder_path = os.path.join(save_directory, "morph_encoder/")
        self.morph_encoder.save_pretrained(morph_encoder_path, **kwargs)

        #SAVE THE PROJECTION LAYER
        os.mkdir(os.path.join(save_directory, "projection_layer/"))
        projection_layer_path = os.path.join(save_directory, "projection_layer/projection_layer.pt")
        #self.projection_layer.save_pretrained(projection_layer_path, **kwargs, safetensors=True)
        torch.save(self.projection_layer.state_dict(), projection_layer_path)

        return
    



#in use
class MorphModelDataCollator(DataCollatorForSeq2Seq):
    """
    pads such that the lang input_ids and tags are padded to the same sequence length
    which is necessary if concatenating the two encoders along the embedding dimension
    """

    #in use
    def __call__(self, data):

        #print(f"custom data collator input: {data}")
        #print(f"custom data collator input: {[d for d in data]}")

        input_ids = [d['input_ids'] for d in data]
        tags = [d['tags'] for d in data]
        labels = [d['labels'] for d in data]
        
        #new
        max_length_input = max(len(ids) for ids in input_ids)
        max_length_tags = max(len(ids) for ids in tags)
        max_length = max(max_length_input, max_length_tags)
        max_label_length = max(len(ids) for ids in labels)

        batch_size = len(data)
        #print(f"batch size: {batch_size}")
        padded_input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        padded_tags = torch.zeros((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        tags_attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        padded_labels = torch.full((batch_size, max_label_length), -100, dtype=torch.long)

        for i, (input_ids_seq, tags_seq, label_seq) in enumerate(zip(input_ids, tags, labels)):
            # Pad input_ids and create attention mask
            padded_input_ids[i, :len(input_ids_seq)] = torch.tensor(input_ids_seq, dtype=torch.long)
            attention_mask[i, :len(input_ids_seq)] = 1

            # Pad new_input_ids and create new attention mask
            padded_tags[i, :len(tags_seq)] = torch.tensor(tags_seq, dtype=torch.long)
            tags_attention_mask[i, :len(tags_seq)] = 1
        
            padded_labels[i, :len(label_seq)] = torch.tensor(label_seq, dtype=torch.long)


        batch = {
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'tags': padded_tags,
            'tags_attention_mask': tags_attention_mask,
            'labels': padded_labels,
        }
        return batch

