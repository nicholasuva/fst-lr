#taking inspiration from this notebook
#https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict, load_from_disk
from typing import Callable
from torch import cuda, device, set_default_device, float16
from evaluate import evaluator
import numpy as np


import psutil
import resource
#some stuff that will be used everywhere perhaps
model_checkpoint = 'jbochi/madlad400-3b-mt'

def create_tokenizer(
        model_checkpoint: str
    ) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    return tokenizer


def create_preprocess_function(
        tokenizer: T5Tokenizer,
        src_lang: str,
        trg_lang: str,
        max_input_length: int = 25,
        max_target_length: int = 25
    ) -> Callable:
    """
    
    """
    def preprocess_function(
            dataset: Dataset
        ):
        """

        """
        prefix = '<2'+trg_lang+'> '
        inputs = [(prefix + ex[src_lang]) for ex in dataset['translation']]
        targets = [(ex[trg_lang]) for ex in dataset['translation']]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        #does the t5tokenizer do this?
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    return preprocess_function




def load_tokenized_inputs(
    tokenizer: T5Tokenizer,
    src_lang: str,
    trg_lang: str,  
    ):
    """
    
    """
    preprocess_function = create_preprocess_function(tokenizer, src_lang, trg_lang)
    datasetdict = load_from_disk(src_lang+'-'+trg_lang+'-combined.hf')
    tokenized_datasets = datasetdict.map(preprocess_function, batched=True)
    return tokenized_datasets



def compute_metrics(
        eval_preds
    ):
    """
    
    """
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in preds]
    decoded_labels = [[label.strip()] for label in labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def finetune_and_eval(
        model_checkpoint,
        src_lang: str,
        trg_lang: str
    ) -> None:
    """
    
    """
    if cuda.is_available():
        this_device="cuda"
        use_fp16=True
    else:
        this_device="cpu"
        use_fp16=False
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, torch_dtype=float16)
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    print('model loaded')
    #batch_size = 16
    batch_size = 1
    args = Seq2SeqTrainingArguments(
        f"{model_checkpoint}-finetuned-{src_lang}-to-{trg_lang}",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=use_fp16,
        push_to_hub=False
    )#kind of copied from whole cloth, need to understand what this means
    args = Seq2SeqTrainingArguments(f"{model_checkpoint}-finetuned-{src_lang}-to-{trg_lang}")
    tokenized_datasets = load_tokenized_inputs(tokenizer, src_lang, trg_lang)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets['train'].train_test_split(test_size=0.0005)['test'],
        eval_dataset=tokenized_datasets['test'].train_test_split(test_size=0.005)['test'], #or should this be dev? I never know ugh
        data_collator = data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    with device(this_device):
        #trainer.evaluate()
        print('about to train')
        trainer.train()
    return

def eval(
        model_checkpoint,
        src_lang: str,
        trg_lang: str
    ) -> None:
    """
    
    """
    try:
        #hacky, rewrite
        test_dataset = load_from_disk(src_lang+'-'+trg_lang+'-combined.hf')
    except:
        test_dataset = load_from_disk(trg_lang+'-'+src_lang+'-combined.hf')
    test_dict = {src_lang:[], trg_lang:[]}
    for pair in test_dataset['test']:
        test_dict[src_lang].append('<2'+trg_lang+'> ' + pair['translation'][src_lang])
        test_dict[trg_lang].append(pair['translation'][trg_lang])
    formatted_ds = Dataset.from_dict(test_dict)

    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    #tokenized_datasets = load_tokenized_inputs(tokenizer, src_lang, trg_lang)
    task_evaluator = evaluator("translation")
    eval_results = task_evaluator.compute(
        model_or_pipeline=model_checkpoint,
        data=formatted_ds.train_test_split(test_size=0.02)['test'],
        input_column=src_lang,
        label_column=trg_lang,
        tokenizer=tokenizer
    )
    print(eval_results)
    return



def main() -> None:
    #model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    #tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    #vocab = tokenizer.get_vocab()
    #print(vocab)
    """
    # Calculate the maximum memory limit (80% of available memory)
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available
    memory_limit = int(available_memory * 0.8)
    
    # Set the memory limit
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    """
    print(cuda.device_count())
    print(cuda.is_available())
    set_default_device("cuda")
    #eval(model_checkpoint, 'se', 'en')
    finetune_and_eval(model_checkpoint, 'en', 'se')
    return

if __name__ == "__main__":
    main()