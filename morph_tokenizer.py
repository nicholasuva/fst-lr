from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from datasets import load_from_disk


def cat_tag_data():
    dataset = []
    return dataset


def create_morph_tokenizer(
        #datasets: list,
        
):
    data = cat_tag_data()
    dataset = load_from_disk('en-fi-combined.hf')
    data = dataset['train']['fi tags']
    print(data[0])
    data = [' '.join(tag for tag in tags) for tags in data]
    data = [sent.replace('+', '') for sent in data]
    print(data[0])
    tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
    tokenizer.train_from_iterator(data, trainer)
    tokenizer.save('morph_tag_tokenizer.json')
    test_sent = data[0]
    encoded = tokenizer.encode(test_sent)
    print(encoded.tokens)
    print(encoded.ids)
    return tokenizer

def main():
    tokenizer = create_tokenizer()
    return

if __name__ == "__main__":
    main()
