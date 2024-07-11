#! /usr/bin/env python

from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from torchview import draw_graph
import torch

model_name = 'jbochi/madlad400-3b-mt'
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_name)

text = "<2it> Good morning!"
text = "<2udm> Hello, my name is Nick, and I am a linguistics student."
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
input_data = tokenizer(text, return_tensors="pt")
#outputs = model.generate(input_ids=input_ids, max_new_tokens=20)

#print(tokenizer.decode(outputs[0], skip_special_tokens=True))

decoder_input_ids = tokenizer("this is a test", return_tensors="pt").input_ids


batch_size = int(2)
input_size = int(128)
#batch_size = torch.ones((2), dtype=torch.int8)
#input_size = torch.ones((128), dtype=torch.int8)
batch_size = torch.tensor(2)
input_size = torch.tensor(128)

#model_graph = draw_graph(model, input_size=(torch.tensor(len(input_data), dtype=int)), device='meta')
#model_graph = draw_graph(model, input_data=input_data, decoder_input_ids=decoder_input_ids, device='meta', save_graph=True, filename='testviz.png')
#model_graph = draw_graph(model, input_size=(2,2), device='meta')
#model_graph.visual_graph


#need to calculate bleu score
def bleu_calc(ref_sent_list, candidate_sent_list):
    if len(ref_sent_list) != len(candidate_sent_list):
        raise ValueError('number of reference and candidate sentences does not match')
    bleu_score = 0.0
    for i in range(len(ref_sent_list)):
        this_ref = ref_sent_list[i].rstrip('\n').split()
        this_candidate = candidate_sent_list[i].rstrip('\n').split()
        bleu_score += sentence_bleu([this_ref], this_candidate, weights=(1.0))
    bleu_score /= len(ref_sent_list)
    return bleu_score



test_refs = [
                "The dog is on the chair.",
                "The cat is on the couch.",
                "The bird is on the hat."
]

test_candidates = [
                "The dog is on chair.",
                "cat is upon this couch.",
                "there is a bird on the hat."
]


test_it = [
                "Il cane sta sulla sedia.",
                "Il gatto sta sul divano.",
                "C'e un ucello sul cappello."
]


text = "<2it> Hello, my name is Nick, and I am a linguistics student."
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
outputs = model.generate(input_ids=input_ids, max_new_tokens=20)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def model_test(sent_list):
    for sent in sent_list:
        text = "<2en> " + sent
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=input_ids, max_new_tokens=20)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return




#test_bleu = bleu_calc(test_refs, test_candidates)
#print(test_bleu)

model_test(test_it)


# Eu adoro pizza!
