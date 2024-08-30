#taking inspiration from this notebook
#https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

from transformers import Seq2SeqTrainer, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict


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
    ) -> function:
    """
    
    """
    def preprocess_function(
            dataset: Dataset
        ) -> whatever:
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
    ) -> something:
    """
    
    """
    preprocess_function = create_preprocess_function(tokenizer, src_lang, trg_lang)
    datasetdict = datasets.load_from_disk('the name of the dataset')
    tokenized_datasets = datasetdict.map(preprocess_function, batched=True)
    return tokenized_datasets



def compute_metrics(
        eval_preds
    ):
    """
    
    """
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
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    tokenizer = T5Tokenizer(model_checkpoint)

    batch_size = 16
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )#kind of copied from whole cloth, need to understand what this means

    tokenized_datasets = load_tokenized_inputs(tokenizer, src_lang, trg_lang)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'], #or should this be dev? I never know ugh
        data_collator = data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return

def main() -> None:
    return

if __name__ == "__main__":
    main()