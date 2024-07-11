#! /usr/bin/env python

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
#from translate.storage.tmx import tmxfile
#from tmx.structural import tmx
#from tmx import load_tmx
from datasets import load_dataset, concatenate_datasets, DatasetDict
from nltk.translate.bleu_score import sentence_bleu
import torch
from torchinfo import summary
print('all imports done')

def load_model():
    model_name = 'jbochi/madlad400-3b-mt'
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    #print(model.get_input_embeddings())
    #print(model.get_output_embeddings())
    print(model.config)
    #print(model)
    summary(model)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

load_model()


def load_model_in_pipeline():
    model_checkpoint = 'jbochi/madlad400-3b-mt'
    translator = pipeline('translation', model=model_checkpoint, max_length=20)
    print(translator("<2it> Hello, my name is Nick, and I am a linguistics student."))
    return


#trying to instantiate appropriate datasets
#using this https://huggingface.co/docs/datasets/en/create_dataset

def gen_from_tmx(filepath):
    print('starting')
    #with open(filepath, 'rb') as source:
        #tmx_file = tmxfile(source, 'en', 'ga')

    tmx_file: tmx = load_tmx(filepath)

    print('file opened')
    #for node in tmx_file.unit_iter():
        #print(node.source, node.target)

    for tu in tmx_file.tus:
        for index, tuv in enumerate(tu.tuvs):
            print(tuv)
        
    return


def gen_from_raw_aligned(src_filename, trg_filename):
    #check num lines are same
    with open(src_filename, 'rb') as src_file:
        src_num_lines = sum(1 for _ in src_file)
    with open(trg_filename, 'r') as trg_file:
        trg_num_lines = sum(1 for _ in trg_file)
    if src_num_lines != trg_num_lines:
        raise ValueError('source and target language files do not have same number of lines')
    


    with open(src_filename, 'r') as src_file:
        with open(trg_filename, 'r') as trg_file:
            pass
            #for src_line, trg_line in 
    return
            
            
        
#i dont think i need to do this i can just get the datasets directly from huggingface

#gen_from_tmx('/home/nicholas/Documents/Thesis/Corpora/NLLB/IrishEnglish/en-ga.tmx')
#if I can just pick all the language pairs and then load them up using the huggingface premade datasets 
# i don't need to do any dataset creation


"""
Looking at OPUS
language            corpora
Norwegian Bokmal    paracrawl kde4 qed wikimedia
Faroese             nllb xlent wikimatrix gnome
Finnish             ccmatrix nllb   paracrawp hplt ccaligned
Northern Sami       kde4    wikimedia   tatoeba
Irish               NLLB paracrawl ccmatrix hplt
Cornish             tatoeba wikimedia   qed


corpora on huggingface: tatoeba paracrawl kde4 
maybe QED? it's listed as QedAmara
It seems like the full NLLB is not up there? only parts



ok so
basically
we take the dataset


"""

def my_load_dataset(dataset, lang1, lang2):
    raw_dataset = load_dataset(dataset, lang1=lang1, lang2=lang2, trust_remote_code=True)
    #print(raw_dataset)
    train_test = raw_dataset['train'].train_test_split(test_size=0.2)
    valid_test = train_test['test'].train_test_split(test_size=0.5)
    three_way_split_ds = DatasetDict({
        'train': train_test['train'],
        'test': valid_test['test'],
        'dev': valid_test['train']
    })
    #print(three_way_split_ds['train'][1]['translation'])
    return three_way_split_ds


def load_combine_save_dataset(dataset_names: list, lang1: str, lang2: str):
    dataset_list = []
    for dataset_name in dataset_names:
        raw_dataset = load_dataset(dataset_name, lang1=lang1, lang2=lang2, split='train', trust_remote_code=True)
        dataset_list.append(raw_dataset)
    if len(dataset_list) > 1:
        for i in range(len(dataset_list)-1):
            assert dataset_list[i].features.type == dataset_list[i+1].features.type
    combined_dataset = concatenate_datasets(dataset_list)

    #print(raw_dataset)
    train_test = combined_dataset.train_test_split(test_size=0.2)
    valid_test = train_test['test'].train_test_split(test_size=0.5)
    three_way_split_ds = DatasetDict({
        'train': train_test['train'],
        'test': valid_test['test'],
        'dev': valid_test['train']
    })
    #print(three_way_split_ds['train'][1]['translation'])

    #save the dataset as I've processed it, locally
    ds_filename = lang1 + '-' + lang2 + '-' + 'combined.hf'
    three_way_split_ds.save_to_disk(ds_filename)
    return three_way_split_ds

def load_or_create_my_dataset(lang1, lang2, dataset_names=[]):
    ds_filename = lang1 + '-' + lang2 + '-' + 'combined.hf'
    #check if it exists locally
    try:
        this_dataset = load_dataset(ds_filename)
    except:
        if len(dataset_names) > 0:
            this_dataset = load_combine_save_dataset(dataset_names, lang1, lang2)
        else:
            raise Exception("must include dataset names to load new dataset")
    return this_dataset

#test_dataset = load_or_create_my_dataset('it', 'en', dataset_names=['flores'])


def single_sent_bleu_eval(ref_sent, candidate_sent):
    split_ref = ref_sent.rstrip('\n').split()
    split_cand = candidate_sent.rstrip('\n').split()
    bleu_score = sentence_bleu([split_ref], split_cand, weights=(1.0, 0.0))
    return bleu_score

#my_load_dataset('kde4', 'en', 'it')


def main():
    ds = my_load_dataset('kde4', 'en', 'it')
    model_checkpoint = 'jbochi/madlad400-3b-mt'
    translator = pipeline('translation', model=model_checkpoint)
    
    total_bleu = 0.0
    sanity_check_data = ds['test'].train_test_split(test_size=0.0004)
    print(sanity_check_data['test'].num_rows)
    total_sents = sanity_check_data['test'].num_rows
    for pair in sanity_check_data['test']:
        print(pair)
        en_sent = pair['translation']['en']
        it_sent = pair['translation']['it']
        it_pred = translator('<2it> ' + en_sent)

        print(it_pred)
        this_score = single_sent_bleu_eval(it_sent, it_pred[0]['translation_text'])
        print('this bleu score: '+str(this_score))
        total_bleu += this_score
    bleu_score = total_bleu / total_sents
    print('bleu:    ' + str(bleu_score))
    return



#print(torch.cuda.is_available())
#main()




#load_model_in_pipeline()
