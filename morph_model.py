from transformers import M2M100ForConditionalGeneration, M2M100Config, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
from torch import cat, float16
import torch
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = M2M100ForConditionalGeneration.from_pretrained('facebook/nllb-200-distilled-600M')
#config = M2M100Config.from_pretrained('facebook/nllb-200-distilled-600M')



class MorphM2M100(M2M100ForConditionalGeneration):
    def __init__(self, config, max_length=200):
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

    def _generate_causal_mask(self, decoder_input_ids):
        seq_len = decoder_input_ids.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=decoder_input_ids.device)).unsqueeze(0).unsqueeze(0)
        return causal_mask

    def forward(
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
                    head_mask=head_mask,
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
    
class MorphModelDataCollator(DataCollatorForSeq2Seq):
    """
    pads such that the lang input_ids and tags are padded to the same sequence length
    which is necessary if concatenating the two encoders along the embedding dimension
    """
    #def __init__(self, tokenizer, model, padding=True):
    #    self.tokenizer = tokenizer
    #    self.model = model
    #    self.padding = padding

    def __call__(self, data):
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