from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, load_from_disk
from nltk.tokenize import word_tokenize
#https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/minhash_deduplication.py
from minhash_deduplication import deduplicate_dataset
from transformers import T5Tokenizer
import sys
import subprocess
from io import TextIOWrapper
from googletrans import Translator
import random
import time

import utils
import hfst
import json
import logging

hf_datasets_list = [
    ''
]

nllb_test = [
    ('aka', 'Akan'),
    ('amh', 'Amharic'),
    ('ara', 'Arabic'),
    ('aym', 'Aymara'),
    ('bak', 'Bashkir'),
    ('bul', 'Bulgarian'),
    ('ces', 'Czech'),
    ('deu', 'German'),
    ('eng', 'English'),
    ('epo', 'Esperanto'),
    ('est-x-plamk', 'Estonian (plamk)'),
    ('est-x-utee', 'Estonian (utee)'),
    ('fao', 'Faroese'),
    ('fin', 'Finnish'),
    ('gle', 'Irish'),
    ('grn', 'Guarani'),
    ('hin', 'Hindi'),
    ('hun', 'Hungarian'),
    ('khk', 'Halh Mongolian'),
    ('lit', 'Lithuanian'),
    ('luo', 'Luo (Kenya and Tanzania)'),
    ('nno', 'Norwegian Nynorsk'),
    ('nno-x-ext-apertium', 'Norwegian Nynorsk (Apertium)'),
    ('nob', 'Norwegian Bokmål'),
    ('ron', 'Romanian'),
    ('rus', 'Russian'),
    ('som', 'Somali'),
    ('sqi', 'Albanian'),
    ('swe', 'Swedish'),
    ('tat', 'Tatar'),
    ('tel', 'Telugu'),
    ('tgl', 'Tagalog'),
    ('tha', 'Thai'),
    ('tir', 'Tigrinya'),
    ('zul-x-exp', 'Zulu')
]

nllb_and_giellalt_low_resource_langs = [
    ('aka', 'Akan'),
    ('amh', 'Amharic'),
    ('aym', 'Aymara'),
    ('bak', 'Bashkir'),
    ('fao', 'Faroese'),
    ('gle', 'Irish'),
    ('grn', 'Guarani'),
    ('luo', 'Luo (Kenya and Tanzania)'),
    ('som', 'Somali'),
    ('tat', 'Tatar'),
    ('tgl', 'Tagalog'),
    ('tir', 'Tigrinya'),
    ('zul-x-exp', 'Zulu')
]


nllb_LR_good_giellalt_covg = [
    ('bak', 'Bashkir'),
    ('fao', 'Faroese'),
    ('gle', 'Irish'),
    ('som', 'Somali'),
    ('tat', 'Tatar'),
]

nllb_LR_good_giellalt_covg_iso1 = ['ba', 'fo', 'ga', 'so', 'tt']

madlad_and_giellalt_but_not_nllb_LR_langs = [
                      ('chr', 'Cherokee'),
                      ('cor', 'Cornish'),
                      ('eus', 'Basque'),
                      ('hil', 'Hiligaynon'),
                      ('iku', 'Inuktitut'),
                      ('kal', 'Kalaallisut'),
                      ('kjh', 'Khakas'),
                      ('koi', 'Komi-Permyak'),
                      ('lav', 'Latvian'),
                      ('mdf', 'Moksha'),
                      ('oji', 'Ojibwa'),
                      ('sme', 'Northern Sami'),
                      ('tyv', 'Tuvinian'),
                      ('udm', 'Udmurt'),
                      ('xal', 'Kalmyk'),
                      ]


corpus_list = [
    'facebook/flores', 
    'Helsinki-NLP/qed_amara', #no
    'Helsinki-NLP/opus-100', #yes
    'Helsinki-NLP/opus_gnome', #no
    'Helsinki-NLP/opus_ubuntu', #no
    'Helsinki-NLP/tatoeba', #no
    'Helsinki-NLP/tatoeba_mt', #no
    'Helsinki-NLP/open_subtitles', #no
    'mteb/NTREX', #what is it? also davidstap/NTREX?
    'ayymen/Weblate-Translations', #not sure what it is, need to look into
    'ayymen/Pontoon-Translations',
    'visheratin/laion-coco-nllb', #need to look
    'bible-nlp/biblenlp-corpus',
    'yhavinga/ccmatrix',
    'nazimali/quran',
]

corpus_dict = {
    'facebook/flores': ['ak', 'am', 'ba'], 
    'Helsinki-NLP/qed_amara': ['ak', 'am', 'ay'],
    'Helsinki-NLP/opus-100': ['am'],
    'Helsinki-NLP/opus_gnome': ['am'],
    'Helsinki-NLP/opus_ubuntu': ['ak', 'am'],
    'Helsinki-NLP/tatoeba': ['am'],
    'Helsinki-NLP/tatoeba_mt': [],
    'Helsinki-NLP/open_subtitles': [],
    'mteb/NTREX': ['am'], #what is it? also davidstap/NTREX?
    'ayymen/Weblate-Translations': ['ak', 'am', 'ay'], #not sure what it is: [], need to look into
    'ayymen/Pontoon-Translations': ['am', 'ay'],
    'visheratin/laion-coco-nllb': ['ak'], #need to look
    'bible-nlp/biblenlp-corpus': [],
    'yhavinga/ccmatrix': ['am'],
    'nazimali/quran': []
}


chosen_langs_with_corpora = [
    ('aka', 'Akan'),
    ('amh', 'Amharic'),
    ('aym', 'Aymara'),
    ('bak', 'Bashkir'),
    ('fao', 'Faroese'),
    ('gle', 'Irish'),
    ('grn', 'Guarani'),
    ('luo', 'Luo (Kenya and Tanzania)'),
    ('som', 'Somali'),
    ('tat', 'Tatar'),
    ('tgl', 'Tagalog'),
    ('tir', 'Tigrinya'),
    ('zul-x-exp', 'Zulu')
]



#ideally deduping will make it not a problem to accidentally reuse same or similar corpora
#possibly should just do helsinki-nlp/* since all their corpora are probably formatted the same


#spanish and turkish failed to build hfst taggers properly

#cache implementation from online that I don't think solves my problem
def cached_detect_lang(text, translator):
    if not hasattr(detect_lang, 'cache'):
        detect_lang.cache = {}
    if text in detect_lang.cache:
        return detect_lang.cache[text]
    
    result = translator.detect(text).lang
    detect_lang.cache[text] = result
    return result

#non-cached version, implementing re-trying and pausing for issues of rate limiting 
def detect_lang(text, translator):
    detection = None
    while detection is None:
        #using googletrans library
        try:
            detection = translator.detect(text)
        #time delay to attempt to avoid rate limiting
        except:
            time.sleep(1)
    result = detection.lang
    return result

def confirm_dataset_lang(
        dataset,
        lang, #iso1 code
        num_of_samples=10,
        confirmation_rate = 0.5
):
    if num_of_samples >= len(dataset):
        return False
        #num_of_samples = len(dataset) - 1
    translator = Translator()
    sample_nums = random.sample(range(0,len(dataset)-1), num_of_samples)
    print(sample_nums)
    samples = [str(dataset[f"{lang}_text"][num]) for num in sample_nums]
    samples = [sent for sent in samples if sent != '']
    num_of_samples = len(samples)
    confirm_ct = 0
    for num in sample_nums:

        sent = str(dataset[f"{lang}_text"][num])
        print(sent)
        print(type(sent))
        #time.sleep(1)
        det_lang = detect_lang(sent, translator)
        print(det_lang)
        if det_lang == lang:
            confirm_ct += 1
    confirm_ratio = float(confirm_ct) / float(num_of_samples)
    if confirm_ratio < confirmation_rate:
        return False
    else:
        return True

    



#NOT in use
def dict_from_json(filename):
    with open(filename) as source:
        data = json.load(source)
    return data

#in use, likely needs to be modified
def build_hfst_taggers(lang_code_list):
    codes, names = zip(*lang_code_list)
    for code in codes:
        subprocess.run(["./build_hfst_tagger.sh", code])
    return

#NOT in use
def check_for_tokenizer(lang_code_list):
    codes, names = zip(*lang_code_list)
    for code in codes:
        subprocess.run(["head", "./lang-"+code+"/src/fst/analyser-gt-desc.hfstol", "--lines=0"])
    return




#in use
def load_and_format_flores_ds(
        dataset_path,
        lang1, #expected in 2letter iso format
        lang2
):
    """
    Works with:
        facebook/flores
    flores datasets must be called with one keyword, formatted like 'lang_Script-lang_Script', eg: 'eng_Latn-aka_Latn'
    flores datasets are formatted like this:
    features: ['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink', 'sentence_eng_Latn', 'sentence_aka_Latn']
    I want to format it 
    """
    lang1, lang2 = sorted([lang1, lang2])
    lang1_nllb = utils.get_nllb_code(lang1)
    lang2_nllb = utils.get_nllb_code(lang2)
    dataset_dict = load_dataset(dataset_path, f"{lang1_nllb}-{lang2_nllb}", trust_remote_code=True)

    """
    try:
        dataset_dict = load_dataset(dataset_path, f"{lang1_nllb}-{lang2_nllb}", trust_remote_code=True)
    except:
        dataset_dict = load_dataset(dataset_path, f"{lang2_nllb}-{lang1_nllb}-", trust_remote_code=True)
    """
    all_splits = []
    for split in dataset_dict:
        this_split = dataset_dict[split]
        this_split = this_split.rename_column(f'sentence_{lang1_nllb}', f'{lang1}_text')
        this_split = this_split.rename_column(f'sentence_{lang2_nllb}', f'{lang2}_text')
        this_split = this_split.remove_columns(['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink'])
        all_splits.append(this_split)
    splits_combined = concatenate_datasets(all_splits)
    return splits_combined

#in use
def load_and_format_helsinki_opus_100_ds(
    dataset_path,
    lang1,
    lang2,
):
    """
    Works with:
        Helsinki-NLP/opus-100 (and seemingly only that)
    """
    lang1, lang2 = sorted([lang1, lang2])
    dataset_dict = load_dataset(dataset_path, f"{lang1}-{lang2}", trust_remote_code=True)
    """
    try:
        dataset_dict = load_dataset(dataset_path, f"{lang1}-{lang2}", trust_remote_code=True)
    except:
        dataset_dict = load_dataset(dataset_path, f"{lang2}-{lang1}", trust_remote_code=True)
    """
    all_splits = []
    for split in dataset_dict:
        this_split = dataset_dict[split]
        #print(f"this split: {this_split}")
        lang1_text = [sent[lang1] for sent in this_split['translation']]
        #print(f"lang1 text [0]: {lang1_text[0]}")
        lang2_text = [sent[lang2] for sent in this_split['translation']]
        #print(f"lang2 text [0]: {lang2_text[0]}")
        this_split = this_split.add_column(f'{lang1}_text', lang1_text)
        this_split = this_split.add_column(f'{lang2}_text', lang2_text)
        if 'translation' in this_split.features:
            this_split = this_split.remove_columns(['translation'])
        if 'id' in this_split.features:
            this_split = this_split.remove_columns(['id'])

        #print(f"this split: {this_split}")
        all_splits.append(this_split)
    splits_combined = concatenate_datasets(all_splits)
    return splits_combined


#in use
def load_and_format_helsinki_tatoeba_ds(
    dataset_path,
    lang1,
    lang2,
):
    """
    Works with:
        Helsinki-NLP/tatoeba 
        Helsinki-NLP/qed_amara
        omfg so the call format for load is right but qed amara usess iso 3 instead of 1 for SOME languages but not all why why why why why why why
        so I can just like add some nested try statements I guess, to try and cover the options
        oh my damn lord that is so silly i really cant even believe it oh my godddddddd
        ok I think all of these sites do seem consistent in ordering their lang inputs alphabetically so i will write a fct to do that instead of having try except clauses

    """
    lang1, lang2 = sorted([lang1, lang2])
    lang1_iso3 = utils.get_set3_code(lang1)
    lang2_iso3 = utils.get_set3_code(lang2)
    lang1_query = lang1
    lang2_query = lang2
    try:
        dataset_dict = load_dataset(dataset_path, lang1=lang1, lang2=lang2, trust_remote_code=True)
    except:
        try:
            dataset_dict = load_dataset(dataset_path, lang1=lang1_iso3, lang2=lang2, trust_remote_code=True)
            lang1_query = lang1_iso3
        except:
            try:
                dataset_dict = load_dataset(dataset_path, lang1=lang1, lang2=lang2_iso3, trust_remote_code=True)
                lang2_query = lang2_iso3
            except:
                dataset_dict = load_dataset(dataset_path, lang1=lang1_iso3, lang2=lang2_iso3, trust_remote_code=True)
                lang1_query = lang1_iso3
                lang2_query = lang2_iso3
    """
    try:
        dataset_dict = load_dataset(dataset_path, lang1=lang1, lang2=lang2, trust_remote_code=True)
    except:
        dataset_dict = load_dataset(dataset_path, lang1=lang2, lang2=lang1, trust_remote_code=True)
    """
    all_splits = []
    for split in dataset_dict:
        this_split = dataset_dict[split]
        print(f"this split: {this_split}")
        lang1_text = [sent[lang1_query] for sent in this_split['translation']]
        print(f"lang1 text [0]: {lang1_text[0]}")
        lang2_text = [sent[lang2_query] for sent in this_split['translation']]
        print(f"lang2 text [0]: {lang2_text[0]}")
        this_split = this_split.add_column(f'{lang1}_text', lang1_text)
        this_split = this_split.add_column(f'{lang2}_text', lang2_text)
        this_split = this_split.remove_columns(['translation', 'id'])
        print(f"this split: {this_split}")
        all_splits.append(this_split)
    splits_combined = concatenate_datasets(all_splits)
    return splits_combined

    

#NOT in use
def manually_check_dataset_validity(
        lang1,
        lang2,
        corpus_list #should be a tuple of hf datasets corpus path and loading schema
):
    return

#in use, probably needs serious modifications
def load_datasets(
        dataset_name_list: list[str],
        lang1: str,
        lang2: str,
        sink: TextIOWrapper
        ) -> list[Dataset]:
    """
    takes in a list of dataset names
    corresponding to HuggingFace repo datasets
    and a pair of languages
    loads the datasets
    adds a column of token counts to each dataset for logging
    adds a column of each sentence pair concatenated to each dataset for ease of deduplication 
    returns a list of datasets
    containing parallel text in those languages
    """
    print('Loading datasets...')
    dataset_list = []
    
    ds_loading_list = [
        ('facebook/flores', load_and_format_flores_ds),
        ('Helsinki-NLP/opus-100', load_and_format_helsinki_opus_100_ds),
        ('Helsinki-NLP/tatoeba', load_and_format_helsinki_tatoeba_ds),
        ('Helsinki-NLP/qed_amara', load_and_format_flores_ds),
    ]


    for ds_name, ds_load_fct in ds_loading_list:
        this_dataset = None
        try:
            this_dataset = ds_load_fct(ds_name, lang1=lang1, lang2=lang2)
        except:
            sink.write(f"unable to load dataset {ds_name} for pair {lang1} {lang2}\n\n")
        if this_dataset is not None:
            if confirm_dataset_lang(this_dataset, lang1) and confirm_dataset_lang(this_dataset, lang2):
                sink.write(f"dataset language ID confirmed for {ds_name} for pair {lang1} {lang2}\n\n")
                this_dataset = create_tokens_per_sentence_feature(this_dataset, lang1, lang2)
                this_dataset = create_content_feature(this_dataset, lang1, lang2)
                dataset_log_title = ds_name+'_'+lang1+'_'+lang2
                log_dataset_info(dataset_log_title, this_dataset, lang1, lang2, sink)
                dataset_list.append(this_dataset)
            else:
                sink.write(f"Dataset language ID not correct for {ds_name} for pair {lang1} {lang2}\n\n")
        

    """
    dataset_name = 'facebook/flores'
    try:
        this_dataset = load_and_format_flores_ds(dataset_name, lang1=lang1, lang2=lang2)
        this_dataset = create_tokens_per_sentence_feature(this_dataset, lang1, lang2)
        this_dataset = create_content_feature(this_dataset, lang1, lang2)
        dataset_log_title = dataset_name+'_'+lang1+'_'+lang2
        log_dataset_info(dataset_log_title, this_dataset, lang1, lang2, sink)
        dataset_list.append(this_dataset)
    except:
        sink.write(f"unable to load dataset {dataset_name} for pair {lang1} {lang2}\n\n")

    #OPUS-100
    dataset_name = 'Helsinki-NLP/opus-100'
    try:
        this_dataset = load_and_format_helsinki_opus_100_ds(dataset_name, lang1=lang1, lang2=lang2)
        this_dataset = create_tokens_per_sentence_feature(this_dataset, lang1, lang2)
        this_dataset = create_content_feature(this_dataset, lang1, lang2)
        dataset_log_title = dataset_name+'_'+lang1+'_'+lang2
        log_dataset_info(dataset_log_title, this_dataset, lang1, lang2, sink)
        dataset_list.append(this_dataset)
    except:
        sink.write(f"unable to load dataset {dataset_name} for pair {lang1} {lang2}\n\n")

    #TATOEBA
    dataset_name = 'Helsinki-NLP/tatoeba'
    try:
        this_dataset = load_and_format_helsinki_tatoeba_ds(dataset_name, lang1=lang1, lang2=lang2)
        this_dataset = create_tokens_per_sentence_feature(this_dataset, lang1, lang2)
        this_dataset = create_content_feature(this_dataset, lang1, lang2)
        dataset_log_title = dataset_name+'_'+lang1+'_'+lang2
        log_dataset_info(dataset_log_title, this_dataset, lang1, lang2, sink)
        dataset_list.append(this_dataset)
    except:
        sink.write(f"unable to load dataset {dataset_name} for pair {lang1} {lang2}\n\n")

    #qed amara
    dataset_name = 'Helsinki-NLP/qed_amara'
    try:
        this_dataset = load_and_format_helsinki_tatoeba_ds(dataset_name, lang1=lang1, lang2=lang2)
        this_dataset = create_tokens_per_sentence_feature(this_dataset, lang1, lang2)
        this_dataset = create_content_feature(this_dataset, lang1, lang2)
        dataset_log_title = dataset_name+'_'+lang1+'_'+lang2
        log_dataset_info(dataset_log_title, this_dataset, lang1, lang2, sink)
        dataset_list.append(this_dataset)
    except:
        sink.write(f"unable to load dataset {dataset_name} for pair {lang1} {lang2}\n\n")
    """
        
    print('Datasets loaded.')
    return dataset_list

#NOT in use
def legacy_load_datasets(
        dataset_name_list: list[str],
        lang1: str,
        lang2: str,
        sink: TextIOWrapper
        ) -> list[Dataset]:
    """
    takes in a list of dataset names
    corresponding to HuggingFace repo datasets
    and a pair of languages
    loads the datasets
    adds a column of token counts to each dataset for logging
    adds a column of each sentence pair concatenated to each dataset for ease of deduplication 
    returns a list of datasets
    containing parallel text in those languages
    """
    print('Loading datasets...')
    dataset_list = []
    for dataset_name in dataset_name_list:
        this_dataset = load_dataset(
            dataset_name,
            lang1=lang1,
            lang2=lang2,
            split="train",
            trust_remote_code=True
            )
        this_dataset = create_tokens_per_sentence_feature(this_dataset, lang1, lang2)
        this_dataset = create_content_feature(this_dataset, lang1, lang2)
        dataset_log_title = dataset_name+'_'+lang1+'_'+lang2
        log_dataset_info(dataset_log_title, this_dataset, lang1, lang2, sink)
        dataset_list.append(this_dataset)
    print('Datasets loaded.')
    return dataset_list

#in use
def create_tokens_per_sentence_feature(
        dataset: Dataset,
        lang1: str,
        lang2: str
    ) -> Dataset:
    """
    takes in a dataset and pair of language names
    adds a feature/column to the dataset containing word token counts for each sentence in the pair
    returns the amended dataset
    """
    #create new feature column
    num_tok_column = []
    #tokenize and log tokens (not saving tokens because T5Tokenizer will be used for training)
    for i in range(len(dataset)):
        #calculate num tokens per sentence
        lang1_num: int = len(word_tokenize(dataset[i][f'{lang1}_text']))
        lang2_num: int = len(word_tokenize(dataset[i][f'{lang2}_text']))
        #add numbers to feature column
        this_num_tok = {lang1: lang1_num, lang2: lang2_num}
        num_tok_column.append(this_num_tok)
    #add new column to dataset
    dataset = dataset.add_column("num_tokens", num_tok_column)
    return dataset

#in use
def create_content_feature(
        dataset: Dataset,
        lang1: str,
        lang2: str
    ) -> Dataset:
    """
    takes in a dataset
    adds a feature column to a parallel text dataset
    containing each sentence pair concatenated
    returns the amended dataset
    used for compatibility with the codeparrot minhash deduplication script
    """
    content_column = []
    for pair in dataset:
        this_string = pair[f'{lang1}_text'] + ' ' + pair[f'{lang2}_text']
        content_column.append(this_string)
    dataset = dataset.add_column('content', content_column)
    return dataset


#in use
def log_dataset_info(
        dataset_title: str,
        dataset: Dataset,
        lang1: str,
        lang2: str,
        sink: TextIOWrapper
        ) -> None:
    """
    logs basic information about a dataset
    """
    num_sentence_pairs: int = len(dataset[f'{lang1}_text'])
    lang1_total_tokens: int = 0
    lang2_total_tokens: int = 0
    for pair in dataset:
        lang1_total_tokens += pair['num_tokens'][lang1]
        lang2_total_tokens += pair['num_tokens'][lang2]
    lang1_token_per_sent_avg: float = float(lang1_total_tokens) / float(num_sentence_pairs)    
    lang2_token_per_sent_avg: float = float(lang2_total_tokens) / float(num_sentence_pairs)
    sink.write('Dataset:\t'+dataset_title+'\n')
    sink.write('num sentence pairs:\t'+str(num_sentence_pairs)+'\n')
    sink.write('num total tokens '+lang1+':\t'+str(lang1_total_tokens)+'\n')
    sink.write('num total tokens '+lang2+':\t'+str(lang2_total_tokens)+'\n')
    sink.write('average tokens per sentence '+lang1+':\t'+str(lang1_token_per_sent_avg)+'\n')
    sink.write('average tokens per sentence '+lang2+':\t'+str(lang2_token_per_sent_avg)+'\n')
    sink.write('\n')
    return
    
#in use
def combine_datasets(
        dataset_list: list[Dataset],
        lang1: str,
        lang2: str,
        sink: TextIOWrapper
        ) -> Dataset:
    """
    takes a list of datasets
    of parallel text, all in the same language pairs
    returns a dataset with the combined data of all datasets in list
    """
    print('Combining datasets...')
    if len(dataset_list) > 1:
        for i in range(len(dataset_list)-1):
            assert dataset_list[i].features.type == dataset_list[i+1].features.type
    combined_dataset = concatenate_datasets(dataset_list)
    dataset_log_title = 'combined'+'_'+lang1+'_'+lang2
    log_dataset_info(dataset_log_title, combined_dataset, lang1, lang2, sink)
    print('Datasets combined.')
    return combined_dataset

#in use
def check_for_url(
        sent: str
        ) -> bool:
    """
    checks a string for various web address and programming language related keywords
    """
    url_txt = {'https://', 'www.', 'WWW.', '.com', '.COM', 'javascript', 'Javascript', '@'}
    for item in url_txt:
        if item in sent:
            return True
    return False

#in use
def check_for_length(
        num_toks: int,
        min_len: int = 5,
        max_len: int = 25
        ) -> bool:
    """
    checks a string for a minimum and maximum length
    """
    if num_toks > max_len or num_toks < min_len:
        return True
    return False

#in use
def clean_dataset(dataset: Dataset,
                  lang1: str,
                  lang2: str,
                  sink: TextIOWrapper
                  ) -> Dataset:
    """
    applies the check_for_url() and check_for_length() fcts to clean a dataset
    """
    print('Cleaning dataset...')
    #the IDs of sentence pairs to be discarded
    exclude_rows = set()
    for i in range(len(dataset)):
        lang1_sent = dataset[i][f'{lang1}_text']
        lang2_sent = dataset[i][f'{lang2}_text']
        lang1_num_toks = dataset[i]['num_tokens'][lang1]
        lang2_num_toks = dataset[i]['num_tokens'][lang2]
        if (
            check_for_length(lang1_num_toks) or
            check_for_length(lang2_num_toks) or
            check_for_url(lang1_sent) or
            check_for_url(lang2_sent)
        ):
            exclude_rows.add(i)
    #removing excluded rows
    dataset = dataset.select(
        i for i in range(len(dataset))
        if i not in exclude_rows
    )
    dataset_log_title = 'cleaned'+'_'+lang1+'_'+lang2
    log_dataset_info(dataset_log_title, dataset, lang1, lang2, sink)
    print('Dataset cleaned.')
    return dataset

#in use
def my_dedupe_dataset(
                dataset: Dataset,
                lang1: str,
                lang2: str,
                sink: TextIOWrapper
                ) -> Dataset:
    """
    wrapper around code parrot minhash deduplicate
    to add my logging function
    """
    print('Deduplicating dataset...')
    #looking at jaccard similarity from this paper
    #https://aclanthology.org/2022.acl-long.577.pdf
    dataset, _ = deduplicate_dataset(dataset, jaccard_threshold=0.85)
    dataset_log_title = 'deduped'+'_'+lang1+'_'+lang2
    log_dataset_info(dataset_log_title, dataset, lang1, lang2, sink)
    print('Dataset deduplicated.')
    return dataset

#NOT in use!!!
def stitch_subword_tokens(
        toks: list[str]
        ):
    """
    takes in a list of tokens, tokenized by T5Tokenizer
    such that the start of a new word has an underscore prepended
    returns a list of tokens such that the length is the same as the input
    all subword tokens have been replaced by the entire consituent word
    the leading underscore has been stripped
    the problem is that if I want to assign the morph tag of the full word
    to each of the constituent subword tokens
    I need to reconstitute the word
    TODOneed to test this
    """
    surface_form_toks = []
    buffer = ''
    buffer_ct = 1
    for i in range(len(toks)):
        #if the token is the start of a full word
        if toks[i][0] == '_':
            if i != 0:
                for j in range(buffer_ct):
                    surface_form_toks.append(buffer[1:])
            buffer = toks[i]
            buffer_ct = 1
        else:
            buffer_ct += 1
            buffer = buffer + toks[i]
        if i == len(toks) - 1:
            for j in range(buffer_ct):
                surface_form_toks.append(buffer[1:])
    assert len(toks) == len(surface_form_toks)
    return surface_form_toks


#NOT in use!!!
def add_morph_tags_to_sentence(
        toks: list[str],
        morph_dict: dict,
        ) -> list[str]:
    """
    takes in a sentence and a morphological tag dict in that language
    because T5Tokenizer often tokenizes parts of words, if a word has been split, 
    this function will assign the tag for the whole word to oops oops oops
    I forgot to stitch the word back together
    first I need to create a second list 
    do i need recursion lol
    todoreplace the placeholder with singular noun tag
    9/14/24 I think I am going to deprecate this
    """
    tags = []
    for tok in toks:
        if tok in morph_dict:
            tag = morph_dict[tok]
        else:
            tag = 'placeholder'
        tags.append(tag)
    return tags
    

#NOT in use, to be deprecated???? or saved?????
def load_hfst_tokenizer(src_lang):
    print('Loading hfst tokenizer...')
    giellalt_code = utils.get_giellalt_code(src_lang)
    tokenizer_filename = 'lang-' + giellalt_code + '/tools/tokenisers/tokeniser-disamb-gt-desc.pmhfst'
    input_stream = hfst.HfstInputStream(tokenizer_filename)
    transducers = []
    while not(input_stream.is_eof()):
        transducers.append(input_stream.read())
    input_stream.close()
    tokenizer = hfst.PmatchContainer(transducers)
    print('hfst tokenizer loaded')
    return tokenizer

#in use
def load_hfst_tagger(src_lang):
    print('Loading hfst tagger...')
    giellalt_code = utils.get_giellalt_code(src_lang)
    tagger_filename = 'lang-' + giellalt_code + '/src/fst/analyser-gt-desc.hfstol'
    input_stream = hfst.HfstInputStream(tagger_filename)
    transducers = []
    while not(input_stream.is_eof()):
        transducers.append(input_stream.read())
    input_stream.close()
    tagger = transducers[0]
    print('hfst tagger loaded')
    return tagger

#in use
def add_src_morph_tags(dataset: Dataset, src_lang, sink: TextIOWrapper):
    #hfst_tokenizer = load_hfst_tokenizer(src_lang)
    hfst_tagger = load_hfst_tagger(src_lang)
    total_tokens = 0
    total_unrecognized_tokens = 0
    all_tokens = []
    all_tags = []
    tag_freqs = {}
    total_num_sentences = len(dataset)
    progress_counter = 0
    for sentence in dataset[f'{src_lang}_text']:
        progress_counter += 1
        print('tagging sentences:\t'+str(progress_counter)+'/'+str(total_num_sentences), end='\r')
        #toks = hfst_tokenizer.tokenize(sentence[src_lang])
        toks = word_tokenize(sentence)
        all_tokens.append(toks)
        total_tokens += len(toks)
        tags = []
        for tok in toks:
            #tag = hfst_tagger.lookup(tok)[0][0].split('+', 1)[1]
            tag = hfst_tagger.lookup(tok)
            try:
                clean_tags = []
                this_tok_raw_tags = tag[0][0].split('#')
                for raw_tag in this_tok_raw_tags:
                    clean_tag = raw_tag.split('+', 1)[1]
                    clean_tags.append(clean_tag)
            except:
                clean_tags = ["UNK"]
            for tag in clean_tags:
                tags.append(tag)
                if tag in tag_freqs:
                    tag_freqs[tag] += 1
                else:
                    tag_freqs[tag] = 1
            if clean_tags[0] == "UNK":
                total_unrecognized_tokens += 1
        all_tags.append(tags)
        #print(sentence)
        #print(toks)
        #print(tags)
    #print('\n')
    print('about to add columns to dataset')
    dataset = dataset.add_column(src_lang+' tokens', all_tokens)
    dataset = dataset.add_column(src_lang+' tags', all_tags)
    print('columns added')
    sink.write('HFST Tokenization and Tagging ---------------------\n')
    sink.write('total number of tokens:\t'+str(total_tokens)+'\n')
    sink.write('total number of unrecognized tokens:\t'+str(total_unrecognized_tokens)+'\n')
    tokenizer_covg: float = 1.0 - float(total_unrecognized_tokens)/float(total_tokens)
    #sink.write('overall coverage of recognized tokens:\t'+str(1.0 - float(total_unrecognized_tokens)/float(total_tokens))+'\n')
    sink.write(f'overall coverage of recognized tokens:\t{tokenizer_covg:.2f}\n')
    tag_freqs = {k: v for k, v in sorted(tag_freqs.items(), key=lambda item: item[1])}
    sink.write('tag frequencies\n')
    for key in tag_freqs:
        sink.write(key+':\t'+str(tag_freqs[key])+'\n')
    return dataset

#in use
def split_dataset(
        dataset: Dataset,
        lang1: str,
        lang2: str,
        sink: TextIOWrapper,
        train_size: int = 0.8,
        test_size: int = 0.1,
        dev_size: int = 0.1
        ) -> DatasetDict:
    """
    splits a dataset into train test and dev splits
    returns a DatasetDict
    """
    print('Splitting dataset...')
    #calculate split sizes
    assert train_size + test_size + dev_size == 1.0
    test_split_input = 1.0 - train_size
    valid_split_input = test_size * (1.0 / test_split_input)

    train_test = dataset.train_test_split(test_size=test_split_input)
    valid_test = train_test['test'].train_test_split(test_size=valid_split_input)
    three_way_split_ds = DatasetDict({
        'train': train_test['train'],
        'test': valid_test['test'],
        'dev': valid_test['train']
    })
    #log the split info
    sink.write('split information:\n--------\n')
    sink.write('train proportion:\t'+str(train_size)+'\n')
    sink.write('train num sentence pairs:\t'+str(len(three_way_split_ds['train']))+'\n')
    sink.write('test proportion:\t'+str(test_size)+'\n')
    sink.write('test num sentence pairs:\t'+str(len(three_way_split_ds['test']))+'\n')
    sink.write('dev proportion:\t'+str(dev_size)+'\n')
    sink.write('dev num sentence pairs:\t'+str(len(three_way_split_ds['dev']))+'\n')
    print('Dataset split.')
    return three_way_split_ds

#in use
def remove_extra_columns(
        dataset: Dataset
) -> Dataset:
    """
    removes extra columns added for preprocessing
    """
    dataset = dataset.remove_columns(['num_tokens', 'content'])
    return dataset
    
#in use
def save_dataset_dict(
        dataset_dict: DatasetDict,
        lang1: str,
        lang2: str,
        ) -> None: 
    """
    saves a DatasetDict locally
    """
    print('Saving dataset dict...')
    #save the dataset as I've processed it, locally
    ds_filename = lang1 + '-' + lang2 + '-' + 'combined.hf'
    dataset_dict.save_to_disk(ds_filename)
    print('Dataset Dict saved locally:\t' + ds_filename)
    return

#in use
def preproc_dataset(
        dataset_name_list: list[str],
        lang1: str,
        lang2: str,
        ) -> DatasetDict:
    """
    preprocessing pipeline to download, clean, deduplicate and split datasets
    and save them
    """
    log_filename = lang1 + '_' + lang2 + '_' + 'dataset_preproc_log.txt'
    with open(log_filename, 'w') as sink:
        sink.write('language 1:\t'+lang1+'\n')
        sink.write('language 2:\t'+lang2+'\n')
        sink.write('datasets:\t' + ' '.join([db for db in dataset_name_list])+'\n\n')
        dataset_list = load_datasets(dataset_name_list, lang1, lang2, sink)
        if len(dataset_list)>0:
            dataset = combine_datasets(dataset_list, lang1, lang2, sink)
            dataset = clean_dataset(dataset, lang1, lang2, sink)
            dataset = my_dedupe_dataset(dataset,lang1, lang2, sink)
            dataset = add_src_morph_tags(dataset, lang2, sink)
            dataset = remove_extra_columns(dataset)
            split_ds_dict = split_dataset(dataset, lang1, lang2, sink)
            save_dataset_dict(split_ds_dict, lang1, lang2)
        else: 
            split_ds_dict = None
    return split_ds_dict


def main():



    #split_ds_dict = preproc_dataset(['tatoeba', 'kde4'], 'en', 'se')
    #save_dataset_dict(split_ds_dict, 'en', 'se')

    #test_dataset = load_from_disk('en-fi-combined.hf')['test']
    #with open('morph_test_log.txt', 'w') as sink:
        #test_dataset = add_src_morph_tags(test_dataset, 'fi', sink)
    #return
    #build_hfst_taggers(nllb_test)
    #return


    #testing LR langs for madlad
    #codes, names = zip(*madlad_and_giellalt_but_not_nllb_LR_langs)
    #build_hfst_taggers(madlad_and_giellalt_but_not_nllb_LR_langs)
    #madlad_list = ['kw', 'eu', 'iu', 'kl', 'kv', 'lv', 'oj', 'se']
    #crashes on kl
    #kv dataset too small, need to find other corpora
    #build_hfst_taggers([('sme', 'Northern Sami')])

    #madlad_list = ['lv', 'oj', 'se']
    madlad_list = ['se']
    #for lang in madlad_list:
        #ds = preproc_dataset([], 'en', lang)
    #return
    madlad_high_hfst_covg = ['kw', 'kl', 'kv', 'lv',  'se', 'kjh', 'mdf', 'udm']

    madlad_iso3_that_dont_have_iso1 = ['chr', 'hil', 'kjh', 'mdf', 'tyv', 'udm', 'xal']
    #cherokee and khj also break bc ds is too small
    #madlad_iso3_that_dont_have_iso1 = ['udm', 'xal']
    nllb_LR_good_giellalt_covg_iso1_unfinished_lang_detect = ['so', 'tt', 'zu']
    #this means I need a fct that gets giellalt codes for ones that dont have iso1?
    #for lang in madlad_list:
        #ds = preproc_dataset([], 'en', lang)
    for lang in nllb_LR_good_giellalt_covg_iso1_unfinished_lang_detect:
        ds = preproc_dataset([], 'en', lang)
    return


    #ak_test = preproc_dataset([], 'en', 'ak')
    #need to figure out situation with iso set 1 code for luo
    test_lang_list = ['ak', 'am', 'ay', 'ba', 'fo', 'ga', 'gn', 'so', 'tt', 'tl', 'ti', 'zu']
    for lang in test_lang_list:
        ds = preproc_dataset([], 'en', lang)
    return


    args1 = {
        'dataset_name_list': ['tatoeba', 'kde4'],
        'lang1': 'en',
        'lang2': 'fi'
    }
    split_ds_dict = preproc_dataset(**args1)
    return



    split_ds_dict = preproc_dataset(['tatoeba', 'kde4'], 'en', 'fi')
    #save_dataset_dict(split_ds_dict, 'en', 'fi')
    return


if __name__ == "__main__":
    main()