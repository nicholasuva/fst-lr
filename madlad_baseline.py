#! /usr/bin/env python

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
#from translate.storage.tmx import tmxfile
#from tmx.structural import tmx
#from tmx import load_tmx
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from evaluate import evaluator
from nltk.translate.bleu_score import sentence_bleu
#import torch
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

#load_model()


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

#note to self, I think  when I am combining text from different datasets I should perhaps do preprocessing first, given that I may want to say how many sentences are from which sources
#I may need to do this differently
#I will now look for good methods of dataset preprocessing for translation corpora
#some general good things to do: throw out short sentences, throw out long sentences, throw out urls,
#throw out sentences where there is a very big gap / mismatch in the length of the sequences in the two languages
# I think I will just write and run my own preprocessing thingy
#can I simply delete an elem from a dataset in situ or do i need to copy it over to another dataset?
# I think it is notgoing to work in situ we have to make a new one

def check_for_url(sent: str):
    url_txt = {'https://', 'www.', 'WWW.', '.com', '.COM', 'javascript', 'Javascript', '@'}
    for item in url_txt:
        if item in sent:
            return True
    return False

def check_for_length(sent: str, min_len=5, max_len=25):
    #this is hacky
    space_ct = 0
    for char in sent:
        if char == ' ':
            space_ct += 1
    if space_ct < min_len or space_ct > max_len:
        return True
    return False



def clean_dataset(dataset):
    #the IDs of sentence pairs to be discarded
    exclude_ids = set()
    for pair in dataset:
        pass
    return
        


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

def load_or_create_my_dataset(lang1: str, lang2: str, dataset_names=[]):
    ds_filename = lang1 + '-' + lang2 + '-' + 'combined.hf'
    #check if the combined dataset already exists locally
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

def my_evaluate(
        model,
        data
):
    task_evaluator = evaluator("translation")
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        data=data,
        input_column='en',
        label_column='it'
        #metric='BLEU'?????
    )
    print(eval_results)
    return


#what do I really need to actually get this fully running?
#probably these things 
#1 data loader
#2 evaluator
#something?
#tbh I should really get some better preprocessing pipeline stuff too
#how  to save dataset locally




def main():
    ds = my_load_dataset('kde4', 'en', 'it')
    print(ds)
    for pair in ds['train']:
        #if check_for_length(pair['translation']['en']) or check_for_length(pair['translation']['it']):
        if check_for_url(pair['translation']['en']) or check_for_url(pair['translation']['it']):
            print(pair)
        #print(pair)
    return
    sanity_check_data = ds['test'].train_test_split(test_size=0.0004)
    print(type(sanity_check_data))
    print(sanity_check_data)
    print(type(sanity_check_data['test']))
    print(sanity_check_data['test'])
    scds = sanity_check_data['test']
    print('num test sents:\t'+str(sanity_check_data['test'].num_rows))
    model_checkpoint = 'jbochi/madlad400-3b-mt'
    translator = pipeline('translation', model=model_checkpoint)
    total_bleu = 0.0
    
    total_sents = sanity_check_data['test'].num_rows

    #just testing this eval fct
    test_dict = {'en':[], 'it':[]}
    for pair in scds:
        test_dict['en'].append('<2it> ' + pair['translation']['en'])
        test_dict['it'].append(pair['translation']['it'])
    test_ds = Dataset.from_dict(test_dict)
    my_evaluate(translator, test_ds)
    return
    #end of testing eval fct


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

if __name__ == "__main__":
    main()


#load_model_in_pipeline()
