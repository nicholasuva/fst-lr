from transformers import M2M100ForConditionalGeneration, M2M100Config, Seq2SeqTrainer
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch import cat
import torch
import utils

model = M2M100ForConditionalGeneration.from_pretrained('facebook/nllb-200-distilled-600M')
config = M2M100Config.from_pretrained('facebook/nllb-200-distilled-600M')


class MorphM2M100(M2M100ForConditionalGeneration):
    def __init__(self, config):
        print('before model initialized')
        utils.log_memory_usage()
        super().__init__(config)
        print('after base model initialized')
        utils.log_memory_usage()
        #morph tag encoder
        #self.morph_encoder = self.model.encoder.__class__(config)
        #or should it be this?
        self.morph_encoder = M2M100ForConditionalGeneration(config).get_encoder()
        print('after morph encoder initialized')
        utils.log_memory_usage()
        self.morph_encoder.layers = self.morph_encoder.layers[:1]
        print('after base model reshaped')
        utils.log_memory_usage()
        self.projection_layer = torch.nn.Linear(2048, config.d_model)
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
                labels=None,
                encoder_outputs=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                use_cache=None,
                **kwargs
    ):
          
        
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
            morph_encoder_outputs = self.morph_encoder(
                input_ids=tags,
                attention_mask=attention_mask,
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
            del encoder_outputs
            del morph_encoder_outputs
            #print('combined')
            projected_encoder_outputs = self.projection_layer(combined_encoder_outputs)
            print(f"projected_encoder_outputs shape: {projected_encoder_outputs.shape}")
            print('after projection')
            utils.log_memory_usage()
            del combined_encoder_outputs
        else:
            if isinstance(encoder_outputs, tuple):
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            projected_encoder_outputs = encoder_hidden_states
        


        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = labels[:, :-1]
        if decoder_input_ids is not None and decoder_attention_mask is None:
            decoder_attention_mask = self._generate_causal_mask(decoder_input_ids)



        
        
        print('after decoder attention mask generated')
        utils.log_memory_usage()

        print(f"input ids shape: {input_ids.size() if input_ids is not None else None}")
        print(f"tags shape: {tags.size() if tags is not None else None}")
        print(f"labels shape: {labels.size() if labels is not None else None}")
        
        #print(f"morph_encoder_outputs shape: {morph_encoder_outputs.size()}")
        #print(f"combined_encoder_outputs shape: {combined_encoder_outputs.size()}")
        #print(f"projected_encoder_outputs shape: {projected_encoder_outputs.size()}")
        print(f"decoder_input_ids shape: {decoder_input_ids.size() if decoder_input_ids is not None else None}")
        #print(f"encoder_hidden_states shape: {projected_encoder_outputs.size()}")
        print(f"attention_mask shape: {attention_mask.size() if attention_mask is not None else None}")


        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=projected_encoder_outputs,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            **kwargs
        )

        if isinstance(decoder_outputs, tuple):
            decoder_hidden_states = decoder_outputs[0]
        else:
            decoder_hidden_states = decoder_outputs.last_hidden_state
        print(f"decoder_hidden_states shape: {decoder_hidden_states.shape}")
        print('after decoder pass')
        utils.log_memory_usage()
        del projected_encoder_outputs
        logits = decoder_hidden_states
        del decoder_outputs
        if labels is not None:
            print(f"logits shape: {logits.size()}")
            print(f"labels shape: {labels.size()}")
            #shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            #print(f"shift_logits shape: {shift_logits.size()}")
            print(f"shift_labels shape: {shift_labels.size()}")
            assert logits.size(1) == shift_labels.size(1), f"logits and labels size mismatch!"
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            print('after loss calculated')
            utils.log_memory_usage()
            #memory debugging
            #loss =loss.detach()
            #return {'loss': loss, 'logits': logits}
            return Seq2SeqLMOutput(loss=loss, logits=logits)
        #print('decoder done')
        else:
            return Seq2SeqLMOutput(logits=logits)
    
class MorphModelDataCollator:
    def __init__(self, tokenizer, model, padding=True):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding

    def __call__(self, data):
        input_ids = [d['input_ids'] for d in data]
        tags = [d['tags'] for d in data]
        labels = [d['labels'] for d in data]
        #lang_input_ids = data['input_ids']
        #morph_input_ids = data['tags']
        #labels = data['labels']

        batch_input_ids = self.tokenizer.pad({'input_ids': input_ids}, padding=self.padding, return_tensors='pt')
        batch_labels = self.tokenizer.pad({'input_ids': labels}, padding=self.padding, return_tensors='pt')

        max_length = batch_input_ids['input_ids'].size(1)
        batch_tags = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in tags], batch_first=True, padding_value=0
        )

        if batch_tags.size(1) < max_length:
            padding = torch.zeros((batch_tags.size(0), max_length - batch_tags.size(1)), dtype=torch.long)
            batch_tags = torch.cat((batch_tags, padding), dim=1)

        print(f"batch input ids shape: {batch_input_ids['input_ids'].shape}")
        print(f"batch tags shape: {batch_tags.shape}")
        print(f"batch labels shape: {batch_labels['input_ids'].shape}")
        print(f"batch attention mask shape: {batch_input_ids['attention_mask'].shape}")

        collated_data = {
                            'input_ids': batch_input_ids['input_ids'].long(),
                            'tags': batch_tags.long(),
                            'labels': batch_labels['input_ids'].long(),
                            'attention_mask': batch_input_ids['attention_mask']
        }
        
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
    

        