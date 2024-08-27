from datasets import load_dataset, concatenate_datasets, Dataset

#https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/minhash_deduplication.py
from minhash_deduplication import deduplicate_dataset

from io import TextIOWrapper

def load_datasets(
        dataset_name_list: list[str],
        lang1: str,
        lang2: str
        ) -> list[Dataset]:
    """
    takes a list of dataset names 
    and a pair of languages
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
        dataset_list.append(this_dataset)
    return dataset_list

def combine_datasets(
        dataset_list: list[Dataset],
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
    return combined_dataset

def check_for_url(
        sent: str
        ) -> bool:
    """
    
    """
    url_txt = {'https://', 'www.', 'WWW.', '.com', '.COM', 'javascript', 'Javascript', '@'}
    for item in url_txt:
        if item in sent:
            return True
    return False

def check_for_length(
        sent: str,
        min_len: int = 5,
        max_len: int = 25
        ) -> bool:
    """
    
    """
    #this is hacky
    space_ct = 0
    for char in sent:
        if char == ' ':
            space_ct += 1
    if space_ct < min_len or space_ct > max_len:
        return True
    return False

def clean_dataset(dataset: Dataset,
                  lang1: str,
                  lang2: str
                  ) -> Dataset:
    """
    
    """
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
    #removing excluded rows
    dataset = dataset.select(
        i for i in range(len(dataset))
        if i not in exclude_rows
    )
    return dataset



def split_dataset(
        dataset: Dataset,
        lang1: str,
        lang2: str,
        train_size: int = 0.8,
        test_size: int = 0.1,
        dev_size: int = 0.1
        ) -> Dataset:
    """
    
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
    return three_way_split_ds

def save_dataset(
        dataset: Dataset,
        lang1: str,
        lang2: str,
        ) -> None: 
    #save the dataset as I've processed it, locally
    ds_filename = lang1 + '-' + lang2 + '-' + 'combined.hf'
    dataset.save_to_disk(ds_filename)
    print('Dataset saved locally:\t' + ds_filename)
    return

def preproc_dataset(
        dataset_name_list: list[str],
        lang1: str,
        lang2: str,
        ) -> Dataset:
    """
    
    """
    log_filename = lang1 + '_' + lang2 + '_' + 'dataset_preproc_log.txt'
    with open(log_filename, 'w') as sink:
        sink.write('language 1:\t'+lang1)
        sink.write('language 2:\t'+lang2)
        sink.write('datasets:\t' + ' '.join([db for db in dataset_name_list]))
        dataset_list = load_datasets(dataset_name_list, lang1, lang2)
        dataset = combine_datasets(dataset_list, sink)
        dataset = clean_dataset(dataset, lang1, lang2)
        #looking at jaccard similarity from this paper
        #https://aclanthology.org/2022.acl-long.577.pdf
        dataset, _ = deduplicate_dataset(dataset, jaccard_threshold=0.85)
        save_dataset(dataset)
    return dataset


def main():
    with open("test_file.txt", 'w') as sink:
        print(type(sink))
    load_datasets(['kde4'], 'en', 'se')
    return

main()