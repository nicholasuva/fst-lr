from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from nltk.tokenize import word_tokenize
#https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/minhash_deduplication.py
from minhash_deduplication import deduplicate_dataset

from io import TextIOWrapper

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
    return dataset_list

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
        lang1_num: int = len(word_tokenize(dataset[i]['translation'][lang1]))
        lang2_num: int = len(word_tokenize(dataset[i]['translation'][lang2]))
        #add numbers to feature column
        this_num_tok = {lang1: lang1_num, lang2: lang2_num}
        num_tok_column.append(this_num_tok)
    #add new column to dataset
    dataset = dataset.add_column("num_tokens", num_tok_column)
    return dataset

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
        this_string = pair['translation'][lang1] + ' ' + pair['translation'][lang2]
        content_column.append(this_string)
    dataset = dataset.add_column('content', content_column)
    return dataset



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
    num_sentence_pairs: int = len(dataset['translation'])
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
    if len(dataset_list) > 1:
        for i in range(len(dataset_list)-1):
            assert dataset_list[i].features.type == dataset_list[i+1].features.type
    combined_dataset = concatenate_datasets(dataset_list)
    dataset_log_title = 'combined'+'_'+lang1+'_'+lang2
    log_dataset_info(dataset_log_title, combined_dataset, lang1, lang2, sink)
    return combined_dataset

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

def clean_dataset(dataset: Dataset,
                  lang1: str,
                  lang2: str,
                  sink: TextIOWrapper
                  ) -> Dataset:
    """
    applies the check_for_url() and check_for_length() fcts to clean a dataset
    """
    #the IDs of sentence pairs to be discarded
    exclude_rows = set()
    for i in range(len(dataset)):
        lang1_sent = dataset[i]['translation'][lang1]
        lang2_sent = dataset[i]['translation'][lang2]
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
    return dataset

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
    #looking at jaccard similarity from this paper
    #https://aclanthology.org/2022.acl-long.577.pdf
    dataset, _ = deduplicate_dataset(dataset, jaccard_threshold=0.85)
    dataset_log_title = 'deduped'+'_'+lang1+'_'+lang2
    log_dataset_info(dataset_log_title, dataset, lang1, lang2, sink)
    return dataset

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
    return three_way_split_ds

def remove_extra_columns(
        dataset: Dataset
) -> Dataset:
    """
    removes extra columns added for preprocessing
    """
    dataset = dataset.remove_columns(['num_tokens', 'content'])
    return dataset
    

def save_dataset_dict(
        dataset_dict: DatasetDict,
        lang1: str,
        lang2: str,
        ) -> None: 
    """
    saves a DatasetDict locally
    """
    #save the dataset as I've processed it, locally
    ds_filename = lang1 + '-' + lang2 + '-' + 'combined.hf'
    dataset_dict.save_to_disk(ds_filename)
    print('Dataset Dict saved locally:\t' + ds_filename)
    return

def preproc_dataset(
        dataset_name_list: list[str],
        lang1: str,
        lang2: str,
        ) -> DatasetDict:
    """
    preprocessing pipeline to download, clean, deduplicate and split datasets
    """
    log_filename = lang1 + '_' + lang2 + '_' + 'dataset_preproc_log.txt'
    with open(log_filename, 'w') as sink:
        sink.write('language 1:\t'+lang1+'\n')
        sink.write('language 2:\t'+lang2+'\n')
        sink.write('datasets:\t' + ' '.join([db for db in dataset_name_list])+'\n\n')
        dataset_list = load_datasets(dataset_name_list, lang1, lang2, sink)
        dataset = combine_datasets(dataset_list, lang1, lang2, sink)
        dataset = clean_dataset(dataset, lang1, lang2, sink)
        dataset = my_dedupe_dataset(dataset,lang1, lang2, sink)
        dataset = remove_extra_columns(dataset)
        split_ds_dict = split_dataset(dataset, lang1, lang2, sink)
    return split_ds_dict


def main():
    split_ds_dict = preproc_dataset(['tatoeba', 'kde4'], 'en', 'se')
    save_dataset_dict(split_ds_dict, 'en', 'se')
    return


if __name__ == "__main__":
    main()