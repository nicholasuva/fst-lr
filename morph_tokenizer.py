from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors



def cat_tag_data():
    dataset = []
    return dataset


def create_tokenizer(
        datasets: list,
        
):
    data = cat_tag_data()
    tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(vocab_size=???, special_tokens=['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
    tokenizer.train_from_iterator(data, trainer)
    tokenizer.save('morph_tag_tokenizer.json')