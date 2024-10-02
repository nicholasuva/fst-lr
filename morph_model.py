from transformers import M2M100ForConditionalGeneration, M2M100Config
from torch import cat
import torch

model = M2M100ForConditionalGeneration.from_pretrained('facebook/nllb-200-distilled-600M')
config = M2M100Config.from_pretrained('facebook/nllb-200-distilled-600M')


class MorphM2M100(M2M100ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
        #morph tag encoder
        #self.morph_encoder = self.model.encoder.__class__(config)
        #or should it be this?
        self.morph_encoder = M2M100ForConditionalGeneration(config).get_encoder()
        self.morph_encoder.layers = self.morph_encoder.layers[:1]
        self.model.encoder.layers = self.model.encoder.layers[:1]
        self.model.decoder.layers = self.model.decoder.layers[:1]


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
                **kwargs
    ):  
        
        
        #print("hello world")
        lang_encoder_outputs = self.model.encoder(input_ids, attention_mask=attention_mask, **kwargs)
        #print('lang encoder done')
        morph_encoder_outputs = self.morph_encoder(tags, attention_mask=attention_mask, **kwargs)
        #print('morph encoder done')
        #combined_encoder_outputs = self.combine_encoder_outputs(lang_encoder_outputs, morph_encoder_outputs)
        combined_encoder_outputs = cat((lang_encoder_outputs.last_hidden_state, morph_encoder_outputs.last_hidden_state), dim=-1)
        #print('combined')

        if labels is not None:
            decoder_input_ids = labels[:, :-1]
        else:
            decoder_input_ids = None
        
        decoder_attention_mask = self._generate_causal_mask(decoder_input_ids)

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            #combined_encoder_outputs,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=combined_encoder_outputs,
            #labels=labels,
            **kwargs
        )
        if labels is not None:
            logits = decoder_outputs.last_hidden_state
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1000)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        #print('decoder done')
        else:
            return decoder_outputs
    
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
            [torch.tensor(ids) for ids in tags], batch_first=True, padding_value=0
        )

        if batch_tags.size(1) < max_length:
            padding = torch.zeros((batch_tags.size(0), max_length - batch_tags.size(1)))
            batch_tags = torch.cat((batch_tags, padding), dim=1)

        collated_data = {
                            'input_ids': batch_input_ids['input_ids'].long(),
                            'tags': batch_tags.long(),
                            'labels': batch_labels['input_ids'].long(),
                            'attention_mask': batch_input_ids['attention_mask']
        }
        
        return collated_data