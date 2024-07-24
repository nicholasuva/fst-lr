#! /usr/bin/env python

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset, get_dataset_config_names
from evaluate import evaluator
from nltk.translate.bleu_score import sentence_bleu
from torch import cuda, device
from torchinfo import summary
print('all imports done')


preproc_test_dict = {

                            'id': [
                                    1000,
                                    1001,
                                    1002,
                                    1003,
                                    1004,
                                    1005
                            ],
                            'translation': [
                                            {'en': 'Hi, my name is Nick and I am a student.', 'it': 'Ciao mi chiamo Nick e sono uno studente.'},
                                            {'en': 'I like to study linguistics and various other topics.', 'it': 'Mi piace studiare la linguistica e varie altre materie.'},
                                            {'en': 'You can reach me at my email address, which is test@testing.com', 'it': 'sono disponibile al mio indirizzio email, test@testing.com'},
                                            {'en': 'javascript is my favorite language to code in', 'it': 'javascript e la mia lingua preferita per la informatica'},
                                            {'en': 'too short', 'it': 'troppo breve',},
                                            {'en': 'too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long too long', 'it': 'troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo troppo lungo'}
                                        ]
}


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


Looking at Huggingface
language            corpora
nb Norwegian Bokmal    paracrawl kde4 qed wikimedia
fo Faroese             nllb xlent wikimatrix gnome
fi Finnish             ccmatrix nllb   paracrawp hplt ccaligned
se Northern Sami       kde4    wikimedia   tatoeba
ga Irish               NLLB paracrawl ccmatrix hplt
kw Cornish             tatoeba wikimedia   qed


ok so
basically
we take the dataset


"""
test_ds_list = [
                ['tatoeba', 'nb', 'en'],
                ['tatoeba', 'fo', 'en'],
                ['tatoeba', 'fi', 'en'],
                ['tatoeba', 'se', 'en'],
                ['tatoeba', 'ga', 'en'],
                ['tatoeba', 'kw', 'en'],
                ['opus_paracrawl', 'nb', 'en'],
                ['opus_paracrawl', 'fo', 'en'],
                ['opus_paracrawl', 'fi', 'en'],
                ['opus_paracrawl', 'se', 'en'],
                ['opus_paracrawl', 'ga', 'en'],
                ['opus_paracrawl', 'kw', 'en'],
                ['kde4', 'nb', 'en'],
                ['kde4', 'fo', 'en'],
                ['kde4', 'fi', 'en'],
                ['kde4', 'se', 'en'],
                ['kde4', 'ga', 'en'],
                ['kde4', 'kw', 'en'],
                ['qed_amara', 'nb', 'en'],
                ['qed_amara', 'fo', 'en'],
                ['qed_amara', 'fi', 'en'],
                ['qed_amara', 'se', 'en'],
                ['qed_amara', 'ga', 'en'],
                ['qed_amara', 'kw', 'en']
]
"""
,    
                ['flores', 'nb', 'en'],
                ['flores', 'fo', 'en'],
                ['facebook/flores', 'fin_Latn', 'eng_Latn'],
                ['flores', 'se', 'en'],
                ['flores', 'ga', 'en'],
                ['flores', 'kw', 'en']       
]
"""

usable_databases = {
                    'tatoeba': [
                                ['en', 'nb'],
                                ['en', 'fo'],
                                ['en', 'fi'],
                                ['en', 'se'],
                                ['en', 'ga'],
                                ['en', 'kw']         
                    ],
                    'kde4': [
                                ['en', 'nb'],
                                ['en', 'fi'],
                                ['en', 'se'],
                                ['en', 'ga']      
                    ],
                    'qed_amara': [
                                ['en', 'nb'],
                                ['en', 'fo'],
                                ['en', 'fi'],
                                ['en', 'ga'],
                                ['en', 'kw']         
                    ]      
}


def test_loading_datasets(dataset_and_lang_pair_list):
    accessible_ds = []
    for this_tuple in dataset_and_lang_pair_list:
        ds_name = this_tuple[0]
        lang1 = this_tuple[2]
        lang2 = this_tuple[1]
        try:
            ds = load_dataset(ds_name, lang1=lang1, lang2=lang2, trust_remote_code=True)
            this_tuple.append(ds.num_rows)
            accessible_ds.append(this_tuple)
            print(ds_name+'\t'+lang1+'\t'+lang2+'\t'+'Success')
        except:
            print(ds_name+'\t'+lang1+'\t'+lang2+'\t'+'Failure')
    for this_tuple in accessible_ds:
        ds_name = this_tuple[0]
        lang1 = this_tuple[2]
        lang2 = this_tuple[1]
        print(ds_name+'\t'+lang1+'\t'+lang2+'\t'+'num_sents:'+'\t'+str(this_tuple[3]))
    return accessible_ds



def my_load_dataset(dataset, lang1, lang2):
    """
    DEPRECATED
    """
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


def my_load_clean_one_dataset(dataset_name: str, lang1: str, lang2: str):
    """
    loads a single dataset, cleans it with simple preprocessing, prints num sentences before and after
    returns cleaned dataset
    """
    raw_dataset = load_dataset(dataset_name, lang1=lang1, lang2=lang2, trust_remote_code=True)
    print('----------------------------\n'+dataset_name+'\n--')
    print('num sentences raw:\t'+str(len(raw_dataset)))
    cleaned_dataset = clean_dataset(raw_dataset, lang1, lang2)
    print('num sentences cleaned:\t'+str(len(cleaned_dataset)))
    return cleaned_dataset

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



def clean_dataset(dataset_dict, lang1, lang2):
    dataset = dataset_dict['train']
    #the IDs of sentence pairs to be discarded
    exclude_rows = set()
    for i in range(len(dataset)):
        lang1_sent = dataset[i]['translation'][lang1]
        lang2_sent = dataset[i]['translation'][lang2]
        if (
            check_for_length(lang1_sent) or
            check_for_length(lang2_sent) or
            check_for_url(lang1_sent) or
            check_for_url(lang2_sent)
        ):
            exclude_rows.add(i)
    dataset = dataset.select(
        i for i in range(len(dataset))
        if i not in exclude_rows
    )
    return dataset
        
def new_combine_and_split_datasets(dataset_list: list, lang1: str, lang2: str, train_size=0.8, test_size=0.1, dev_size=0.1, to_save=True):
    """
    takes in a list of datasets, combines them, then splits into a train test dev split
    saves the dataset locally and returns the dataset
    """
    #calculate split sizes
    assert train_size + test_size + dev_size == 1.0
    test_split_input = 1.0 - train_size
    valid_split_input = test_size * (1.0 / test_split_input)


    if len(dataset_list) > 1:
        for i in range(len(dataset_list)-1):
            assert dataset_list[i].features.type == dataset_list[i+1].features.type
    combined_dataset = concatenate_datasets(dataset_list)
    train_test = combined_dataset.train_test_split(test_size=test_split_input)
    valid_test = train_test['test'].train_test_split(test_size=valid_split_input)
    three_way_split_ds = DatasetDict({
        'train': train_test['train'],
        'test': valid_test['test'],
        'dev': valid_test['train']
    })
    print(three_way_split_ds)
    #save the dataset as I've processed it, locally
    if to_save:
        ds_filename = lang1 + '-' + lang2 + '-' + 'combined.hf'
        three_way_split_ds.save_to_disk(ds_filename)
    return three_way_split_ds


def load_combine_save_dataset(dataset_names: list, lang1: str, lang2: str):
    """
    DEPRECATED
    """
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
        test_dataset,
        lang1,
        lang2
):
    
    test_dict = {lang1:[], lang2:[]}
    for pair in test_dataset:
        test_dict[lang1].append('<2'+lang2+'> ' + pair['translation'][lang1])
        test_dict[lang2].append(pair['translation'][lang2])
    formatted_ds = Dataset.from_dict(test_dict)

    task_evaluator = evaluator("translation")
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        data=formatted_ds,
        input_column=lang1,
        label_column=lang2
        #metric='BLEU'?????
    )
    print(eval_results)
    return eval_results


#what do I really need to actually get this fully running?
#probably these things 
#1 data loader
#2 evaluator
#something?
#tbh I should really get some better preprocessing pipeline stuff too
#how  to save dataset locally




def main():
    print(cuda.device_count())
    print(cuda.is_available())
    #return

    #july 24 testing the baseline loading and eval
    se_test_ds = load_dataset('kde4', lang1='en', lang2='se', trust_remote_code=True)
    #print(se_test_ds)
    cleaned = clean_dataset(se_test_ds, 'en', 'se')
    #print(cleaned)
    split_ds = new_combine_and_split_datasets([cleaned], 'en', 'se')
    #print(split_ds)
    with device("cuda"):
        model_checkpoint = 'jbochi/madlad400-3b-mt'
        translator = pipeline('translation', model=model_checkpoint)
        my_evaluate(translator, split_ds['test'], 'en', 'se')
    return



    #testing which datasets to load
    test_loading_datasets(test_ds_list)
    return




    ds = Dataset.from_dict(preproc_test_dict)
    #ds = ds.train_test_split(test_size=0)



    #ds = my_load_dataset('kde4', 'en', 'it')
    print(ds)
    #for pair in ds['train']:
    for pair in ds:
        print(pair)
    ds = clean_dataset(ds, 'en', 'it')
    for pair in ds:
        print(pair)
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




#tue july 23 note - i need to actually sort out my data and my baselines
# so what I need to do is write the fcts for FIRST taking in each ds in each lang pair, 
#THEN logging the info for each
#THEN preprocessing
#THEN logging the info again
#THEN combining them
#THEN logging the info for the combined ds


#wed july 24, I am going to first reimplement my own wrapper for BLEU, then run baseline evals
#need to think about best logging practices
#ideally save to file so I don't have to manually copy
