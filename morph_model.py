from transformers import M2M100ForConditionalGeneration, M2M100Config
from torch import cat

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


    def forward(
                self,
                lang_input_ids,
                morph_input_ids,
                attention_mask=None,
                labels=None,
                **kwargs
    ):  
        
        
        print("hello world")
        lang_encoder_outputs = self.model.encoder(lang_input_ids, attention_mask=attention_mask, **kwargs)
        print('lang encoder done')
        morph_encoder_outputs = self.morph_encoder(morph_input_ids, attention_mask=attention_mask, **kwargs)
        print('morph encoder done')
        #combined_encoder_outputs = self.combine_encoder_outputs(lang_encoder_outputs, morph_encoder_outputs)
        combined_encoder_outputs = cat((lang_encoder_outputs.last_hidden_state, morph_encoder_outputs.last_hidden_state), dim=-1)
        print('combined')

        decoder_outputs = self.model.decoder(
            combined_encoder_outputs,
            attention_mask=attention_mask,
            encoder_hidden_states=combined_encoder_outputs,
            labels=labels,
            **kwargs
        )
        print('decoder done')
        return decoder_outputs