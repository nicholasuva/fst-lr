from transformers import M2M100ForConditionalGeneration, M2M100Config, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutput
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder, shift_tokens_right
from torch import cat, float16
import torch
import utils
from typing import Optional, Tuple, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#class MorphM2M100(torch.nn.Module):
class MorphM2M100(M2M100ForConditionalGeneration):
    #in use
    def __init__(
        self,
        checkpoint,
        morph_encoder_config,
        freeze_base_encoder=False,
        encoder_scheme="embed_dim"
        ):
        """
        possible encoder_scheme values:
            "embed_dim": concatenate encoder inputs along embedding dimension
        """
        #create base encoder-decoder model and lm_head from pretrained checkpoint
        source_model = M2M100ForConditionalGeneration.from_pretrained(checkpoint)
        config = source_model.config
        super().__init__(config)
        self.model = source_model.model
        self.lm_head = source_model.lm_head
        
        #create new encoder for morph tags from config
        self.morph_encoder = M2M100Encoder(morph_encoder_config)

        #create projection layer for correncting concatenated dimension
        if encoder_scheme == "embed_dim":
            self.projection_layer = torch.nn.Linear(
                config.d_model+morph_encoder_config.d_model,
                config.d_model
                )

        #freeze parameters in base model encoder
        if freeze_base_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    #deprecated
    def legacy__init__(self, config, max_length=200):
        print('before model initialized')
        utils.log_memory_usage()
        config.max_length = max_length
        config.torch_dtype=float16
        super().__init__(config)
        print('after base model initialized')
        utils.log_memory_usage()
        #morph tag encoder
        #self.morph_encoder = self.model.encoder.__class__(config)
        #or should it be this?
        morph_encoder_dim = 128
        morph_encoder_config = M2M100Config(
            vocab_size=100,
            #d_model=xxx,
            encoder_layers=2,
            #encoder_attention_heads=4,
            d_model=morph_encoder_dim
        )
        self.morph_encoder = M2M100Encoder(morph_encoder_config)


        #self.morph_encoder = M2M100ForConditionalGeneration(config).get_encoder()
        print('after morph encoder initialized')
        utils.log_memory_usage()
        #self.morph_encoder.layers = self.morph_encoder.layers[:2]
        print('after base model reshaped')
        utils.log_memory_usage()
        self.projection_layer = torch.nn.Linear(config.d_model+morph_encoder_dim, config.d_model)
        print('after projection layer initialized')
        utils.log_memory_usage()
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        #self.model.encoder.layers = self.model.encoder.layers[:1]
        #self.model.decoder.layers = self.model.decoder.layers[:1]


    """
    def combine_encoder_outputs(
        self,
        lang_encoder_outputs,
        morph_encoder_outputs
    ):
        combined_encoder_outputs = cat([lang_encoder_outputs, morph_encoder_outputs], dim=-1)
        return combined_encoder_outputs
    """

    #deprecated
    def _generate_causal_mask(self, decoder_input_ids):
        seq_len = decoder_input_ids.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=decoder_input_ids.device)).unsqueeze(0).unsqueeze(0)
        return causal_mask

    #deprecated
    def wrap_forward(
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
        result = super().forward(
            input_ids=input_ids,
            attention_mask = attention_mask,
            decoder_input_ids= decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return result

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
            encoder_outputs = BaseModelOutput(
                last_hidden_state=projected_encoder_outputs[0],
                hidden_states=projected_encoder_outputs[1] if len(projected_encoder_outputs) > 1 else None,
                attentions=projected_encoder_outputs[2] if len(projected_encoder_outputs) > 2 else None,
            )
            #print(f"modified encoder outputs type: {type(encoder_outputs)}")


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
        #copied from the forward fct from M2M100ForConditionalGeneration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
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
        """
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

    #deprecated
    def legacy_forward(
                self,
                input_ids=None,
                tags=None,
                attention_mask=None,
                tags_attention_mask=None,
                labels=None,
                encoder_outputs=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                use_cache=None,
                decoder_inputs_embeds=None,
                use_morph_encoder=False,
                **kwargs
    ):

        print("-------------start custom forward--------------------")
        print(f"input ids shape: {input_ids.size() if input_ids is not None else None}")
        print(f"tags shape: {tags.size() if tags is not None else None}")
        print(f"labels shape: {labels.size() if labels is not None else None}")
        print(f"attention_mask shape: {attention_mask.size() if attention_mask is not None else None}")
        print(f"tags_attention_mask shape: {tags_attention_mask.size() if tags_attention_mask is not None else None}")


        
        if encoder_outputs is None:
            #print("hello world")
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                **kwargs
                )
            print(f"lang_encoder_outputs shape: {encoder_outputs.last_hidden_state.shape}")
            print('after base encoder pass')
            utils.log_memory_usage()
            #print('lang encoder done')
            if use_morph_encoder:
                morph_encoder_outputs = self.morph_encoder(
                    input_ids=tags,
                    attention_mask=tags_attention_mask,
                    head_mask=head_mask, #do i need this? do I need to make a new one?
                    **kwargs
                    )
                print(f"morph_encoder_outputs shape: {morph_encoder_outputs.last_hidden_state.shape}")
                print('after morph encoder pass')
                utils.log_memory_usage()
                #print('morph encoder done')
                #combined_encoder_outputs = self.combine_encoder_outputs(lang_encoder_outputs, morph_encoder_outputs)
                combined_encoder_outputs = cat((encoder_outputs.last_hidden_state, morph_encoder_outputs.last_hidden_state), dim=-1)
                print(f"combined_encoder_outputs shape: {combined_encoder_outputs.shape}")
                print('after concatenation')
                utils.log_memory_usage()
                #del encoder_outputs
                #del morph_encoder_outputs
                #print('combined')
                projected_encoder_outputs = self.projection_layer(combined_encoder_outputs)
                print(f"projected_encoder_outputs shape: {projected_encoder_outputs.shape}")
                print('after projection')
                utils.log_memory_usage()
                #del combined_encoder_outputs
            else:
                if isinstance(encoder_outputs, tuple):
                    encoder_hidden_states = encoder_outputs[0]
                else:
                    encoder_hidden_states = encoder_outputs.last_hidden_state
                projected_encoder_outputs = encoder_hidden_states
        else:
            if isinstance(encoder_outputs, tuple):
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            projected_encoder_outputs = encoder_hidden_states
        
        #decoder_start_token_id = torch.tensor([256047], device=device, dtype=torch.long)

        if decoder_input_ids is not None:
            print("------decoder input ids passed during eval--------")
            print(f"decoder input ids: {decoder_input_ids}")
            print(f"decoder_input_ids shape: {decoder_input_ids.size()}")


        if (labels is None) and (decoder_input_ids is None) and (decoder_inputs_embeds is None):
            raise ValueError("must call forward with either labels, decoder input ids, or decoder inputs embeds")
        if labels is not None and decoder_input_ids is None:
            print("------decoder input ids created from labels--------")
            print(f"labels: {labels}")
            decoder_input_ids = labels[:, :-1]
            print(f"decoder input ids: {decoder_input_ids}")
            print(f"decoder_input_ids shape: {decoder_input_ids.size() if decoder_input_ids is not None else None}")

            #decoder_input_ids = torch.cat([decoder_start_token_id, decoder_input_ids], dim=-1)
            #print(f"reshaped decoder input ids: {decoder_input_ids}")
            #print(f"reshaped decoder_input_ids shape: {decoder_input_ids.size() if decoder_input_ids is not None else None}")
            print("------end decoder from labels section--------")
        if decoder_input_ids is not None and decoder_attention_mask is None:
            decoder_attention_mask = self._generate_causal_mask(decoder_input_ids)

        if decoder_input_ids is not None:
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                #inputs_embeds=decoder_inputs,
                attention_mask=decoder_attention_mask,
                encoder_attention_mask=attention_mask,
                encoder_hidden_states=projected_encoder_outputs,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                #inputs_embeds=decoder_inputs_embeds,
                **kwargs
            )
        elif decoder_inputs_embeds is not None:
            decoder_outputs = self.model.decoder(
                #input_ids=decoder_input_ids,
                inputs_embeds=decoder_inputs_embeds,
                attention_mask=decoder_attention_mask,
                encoder_attention_mask=attention_mask,
                encoder_hidden_states=projected_encoder_outputs,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                #inputs_embeds=decoder_inputs_embeds,
                **kwargs
            )
        else:
            raise ValueError("Require either decoder_input_ids or decoder_inputs_embeds")



        

        #print(f"morph_encoder_outputs shape: {morph_encoder_outputs.size()}")
        #print(f"combined_encoder_outputs shape: {combined_encoder_outputs.size()}")
        #print(f"projected_encoder_outputs shape: {projected_encoder_outputs.size()}")
        print(f"decoder_input_ids shape: {decoder_input_ids.size() if decoder_input_ids is not None else None}")
        #print(f"encoder_hidden_states shape: {projected_encoder_outputs.size()}")




        if isinstance(decoder_outputs, tuple):
            decoder_hidden_states = decoder_outputs[0]
        else:
            decoder_hidden_states = decoder_outputs.last_hidden_state
        print(f"decoder_hidden_states shape: {decoder_hidden_states.shape}")
        print('after decoder pass')
        utils.log_memory_usage()
        #del projected_encoder_outputs
        logits = self.lm_head(decoder_hidden_states)
        #del decoder_outputs
        #if False:
        if labels is not None:
            print(f"logits shape: {logits.size()}")
            print(f"labels shape: {labels.size()}")
            print(f"labels: {labels}")
            #shift_logits = logits[:, :-1, :].contiguous()
            logits = logits[:,:,:].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            print(f"shift labels: {shift_labels}")
            print(f"logits: {logits}")
            #print(f"shift_logits shape: {shift_logits.size()}")
            print(f"shift_labels shape: {shift_labels.size()}")
            assert logits.size(1) == shift_labels.size(1), f"logits and labels size mismatch!"
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            print(f"type of loss in forward: {type(loss)}")
            print(f"shape of loss in forward: {loss.size()}")
            print(f"loss within forward: {loss}")
            print('after loss calculated')
            utils.log_memory_usage()
            #memory debugging
            #loss =loss.detach()
            #return {'loss': loss, 'logits': logits}
            return Seq2SeqLMOutput(loss=loss, logits=logits)
        #print('decoder done')
        else:
            return Seq2SeqLMOutput(logits=logits)
    
#in use
class MorphModelDataCollator(DataCollatorForSeq2Seq):
    """
    pads such that the lang input_ids and tags are padded to the same sequence length
    which is necessary if concatenating the two encoders along the embedding dimension
    """
    #def __init__(self, tokenizer, model, padding=True):
    #    self.tokenizer = tokenizer
    #    self.model = model
    #    self.padding = padding

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

        batch_size = len(data)
        padded_input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        padded_tags = torch.zeros((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        tags_attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

        for i, (input_ids_seq, tags_seq) in enumerate(zip(input_ids, tags)):
            # Pad input_ids and create attention mask
            padded_input_ids[i, :len(input_ids_seq)] = torch.tensor(input_ids_seq, dtype=torch.long)
            attention_mask[i, :len(input_ids_seq)] = 1

            # Pad new_input_ids and create new attention mask
            padded_tags[i, :len(tags_seq)] = torch.tensor(tags_seq, dtype=torch.long)
            tags_attention_mask[i, :len(tags_seq)] = 1

        batch = super().__call__(data)
        batch['input_ids'] = padded_input_ids
        batch['attention_mask'] = attention_mask
        batch['tags'] = padded_tags
        batch['tags_attention_mask'] = tags_attention_mask
        return batch

        #old, deprecated, to remove
        batch_input_ids = self.tokenizer.pad({'input_ids': input_ids}, padding=self.padding, return_tensors='pt')
        batch_labels = self.tokenizer.pad({'input_ids': labels}, padding=self.padding, return_tensors='pt')

        print(f"batch input ids shape: {batch_input_ids['input_ids'].shape}")
        print(f"batch labels shape: {batch_labels['input_ids'].shape}")
        print(f"batch attention mask shape: {batch_input_ids['attention_mask'].shape}")

        collated_data = {
                            'input_ids': batch_input_ids['input_ids'].long(),
                            'labels': batch_labels['input_ids'].long(),
                            'attention_mask': batch_input_ids['attention_mask']
        }
        
        if isinstance(self.model, MorphM2M100):
            
            max_length = batch_input_ids['input_ids'].size(1)
            batch_tags = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in tags], batch_first=True, padding_value=0
            )
            if batch_tags.size(1) < max_length:
                padding = torch.zeros((batch_tags.size(0), max_length - batch_tags.size(1)), dtype=torch.long)
                batch_tags = torch.cat((batch_tags, padding), dim=1)
            print(f"batch tags shape: {batch_tags.shape}")
            collated_data['tags'] = batch_tags.long()

        return collated_data

#deprecated 
class ForwardOnlyTrainer(Seq2SeqTrainer):
    """
    def training_step(self, model, inputs):
        model.train()
        outputs = model(**inputs)
        loss = outputs['loss']
        print('after forward only pass')
        utils.log_memory_usage()
        return loss
    """
    def training_step(self, model, inputs):
        result = super().training_step(model, inputs)
        print('after training step')
        utils.log_memory_usage()
        return result
    
    def compute_loss(self, model, inputs, return_outputs=False):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs)
        print('after compute_loss')
        utils.log_memory_usage()
        return result

"""
class NewMorphDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # Tokenizer collate function for input_ids (natural language data)
        batch = super().__call__(features)

        # Generate attention mask for tags (for grammatical tags)
        if "tags" in features[0]:
            # Pad tags to max length within the batch
            tags = [f["tags"] for f in features]
            max_length_tags = max(len(ids) for ids in tags)

            # Pad tags and generate new attention_mask for them
            padded_tags = torch.zeros((len(tags), max_length_tags), dtype=torch.long)
            tags_attention_mask = torch.zeros((len(tags), max_length_tags), dtype=torch.long)

            for i, ids in enumerate(tags):
                padded_tags[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                tags_attention_mask[i, :len(ids)] = 1

            # Add tags and tags_attention_mask to the batch
            batch["tags"] = padded_tags
            batch["tags_attention_mask"] = tags_attention_mask

        return batch
"""