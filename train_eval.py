#taking inspiration from this notebook
#https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb
#test test test
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Config, M2M100Tokenizer, AdamW, NllbTokenizer, get_scheduler
from datasets import Dataset, DatasetDict, load_from_disk
from typing import Callable
from torch import cuda, device, set_default_device, Generator, no_grad, bfloat16, float16
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from evaluate import evaluator
import numpy as np
import utils
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
from morph_tokenizer import create_morph_tokenizer

from morph_model import MorphM2M100

import sys
import trace
import psutil
import resource
import hunter
#import function_trace
#some stuff that will be used everywhere perhaps
#model_checkpoint = 'jbochi/madlad400-3b-mt'




def create_preprocess_function(
        tokenizer,
        model_scheme: str, 
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
        prefix = ''
        if model_scheme == 'MADLAD':
            prefix = '<2'+trg_lang+'> '
        if model_scheme == 'NLLB':
            prefix = utils.get_nllb_code(trg_lang) + ' ' #this is wrong I need prefix on the trg lang for nllb ugh wait maybe this is right
        inputs = [(prefix + ex[src_lang]) for ex in dataset['translation']]
        targets = [(ex[trg_lang]) for ex in dataset['translation']]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        #does the t5tokenizer do this?
        if model_scheme == 'MADLAD':
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        if model_scheme == 'NLLB':
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
    preprocess_function = create_preprocess_function(tokenizer, 'NLLB', src_lang, trg_lang)
    datasetdict = load_from_disk(src_lang+'-'+trg_lang+'-combined.hf')
    datasetdict.generator=Generator('cuda')
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
        model_scheme: str,
        src_lang: str,
        trg_lang: str
    ) -> None:
    """
    
    """
    if cuda.is_available():
        this_device="auto"
        use_fp16=True
    else:
        this_device="cpu"
        use_fp16=False
    #with device('cpu'):
    if True:
        #peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules='xxxxxxx')
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, target_modules='all-linear')

        if model_scheme == "MADLAD":
            model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, device_map="auto", torch_dtype=float16)
            model = get_peft_model(model, peft_config)
            tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
            
        if model_scheme == "NLLB":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, device_map="auto")#, torch_dtype=float16)
            #model = get_peft_model(model, peft_config)
            print(next(model.parameters()).is_cuda)
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang=utils.get_nllb_code(src_lang))
        if model_scheme == "Morph":
            config = M2M100Config.from_pretrained('facebook/nllb-200-distilled-600M')
            model = MorphM2M100(config)
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang=utils.get_nllb_code(src_lang))
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
        tokenized_datasets.generator = Generator('cuda')
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_datasets['train'].train_test_split(test_size=0.00005)['test'],
            eval_dataset=tokenized_datasets['test'].train_test_split(test_size=0.005)['test'], #or should this be dev? I never know ugh
            data_collator = data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        #trainer.evaluate()
        print('about to train')
        #tracer = trace.Trace()
        #tracer.run("trainer.train()")
        """
        hunter.trace(
            #stdlib=False,
            #clear_env_var=True,
            #calls=10
            hunter.Q(
                depth_lt=5,
                kind='call',
                stdlib=False,
                #function_startswith=('train')
                #function='train'
                #module='Seq2SeqTrainer',
                #module='transformers.Seq2SeqTrainer'
            ) &
            ~hunter.Q(
                function_startswith=(
                                     '_hp_search_setup',
                                     'n_gpu',
                                     'find_executable',
                                     'free_memory',
                                     'debug',
                                     'world_size',
                                     'has_length',
                                     '__len__',
                                     'num_examples',
                                     'is_sagemaker_mp',
                                     'create_optimizer',
                                     '_wrap',
                                     'get_model',
                                     'is_',
                                     'zero_grad',
                                     'prepare',
                                     'get_train',
                                     '<genexpr>',
                                     'parameters'
                                     )
                #function='train',
                #function='start'
            )
            #module='trainer'
        )
        """
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


def create_model():
    print('creating model')
    initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    #text_tokenizer = xxxxx
    config = M2M100ForConditionalGeneration.from_pretrained(initial_checkpoint).config
    model = MorphM2M100(config)
    if False:
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, target_modules='all-linear')
        model = get_peft_model(model, peft_config)
    print('model created')
    return model

def preproc_data():
    initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    text_tokenizer = NllbTokenizer.from_pretrained(initial_checkpoint)
    tag_tokenizer = create_morph_tokenizer()
    max_input_length = 25
    max_tag_length = 25
    max_target_length = 25
    dataset = load_from_disk('en-fi-combined.hf')
    dataset= dataset['train']
    src_lang = 'fi'
    trg_lang = 'en'
    prefix = utils.get_nllb_code(trg_lang) + ' ' #this is wrong I need prefix on the trg lang for nllb ugh wait maybe this is right
    inputs = [(prefix + ex[src_lang]) for ex in dataset['translation']]
    targets = [(ex[trg_lang]) for ex in dataset['translation']]
    tags_data = [ex for ex in dataset['fi tags']]
    print(tags_data[0])
    print(tags_data[1])
    tags_data = [' '.join(tag for tag in tags) for tags in tags_data]
    tags_data = [sent.replace('+', '') for sent in tags_data]
    #print(tags_data[0])
    #print(tags_data[1])
    model_inputs = text_tokenizer(inputs, max_length=max_input_length, padding=True, truncation=True, return_tensors='pt')
    labels = text_tokenizer(targets, max_length=max_target_length, padding=True, truncation=True, return_tensors='pt')
    model_tag_inputs = [tag_tokenizer.encode(tags).ids for tags in tags_data]
    model_tag_inputs = [enc + [0] * (max_input_length - len(enc)) if len(enc) < max_input_length else enc[:max_input_length] for enc in model_tag_inputs]
    model_tag_inputs = torch.tensor(model_tag_inputs)
    model_inputs['labels'] = labels['input_ids']
    model_inputs['tags'] = model_tag_inputs
    return model_inputs


def create_dataloaders():
    initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    #text_tokenizer = M2M100Tokenizer.from_pretrained(initial_checkpoint)
    #tag_tokenizer = create_morph_tokenizer()
    #dataset = load_from_disk('en-fi-combined.hf')
    data = preproc_data()
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    return dataloader



def morph_custom_train(
        model: MorphM2M100,
        train_dataloader,
        eval_dataloader,
        num_epochs,
        learning_rate=5e-5
    ):

    #ensure correct device
    device = torch.device("cuda") if cuda.is_available() else torch.device("cpu")
    model.to(device)

    #optimizer, scheduler, progress bar
    optimizer=AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    progress_bar = tqdm(range(num_training_steps))

    print('training...')
    for epoch in range(num_epochs):
        #train
        model.train()
        total_training_loss = 0
        for batch in train_dataloader:
            lang_input_ids = batch['lang_input_ids'].to(device)
            morph_input_ids = batch['morph_input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(lang_input_ids=lang_input_ids, morph_input_ids=morph_input_ids, labels=labels)
            loss = outputs.loss
            total_training_loss += loss.item()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        avg_training_loss = total_training_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Avg Training Loss: {avg_training_loss}")
        #eval
        model.eval()
        total_evaluation_loss = 0
        for batch in eval_dataloader:
            with no_grad():
                lang_input_ids = batch['lang_input_ids'].to(device)
                morph_input_ids = batch['morph_input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(lang_input_ids=lang_input_ids, morph_input_ids=morph_input_ids, labels=labels)
                loss = outputs.loss
                total_evaluation_loss += loss.item()
        avg_evaluation_loss = total_evaluation_loss / len(eval_dataloader)
        print(f"Epoch {epoch+1} - Avg Evaluation Loss: {avg_evaluation_loss}")
        #need to like save the model somehow
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

    #finetune_and_eval('facebook/nllb-200-distilled-600M', 'Morph', 'en', 'fi')
    #return


    #model = M2M100ForConditionalGeneration.from_pretrained('facebook/nllb-200-distilled-600m')
    if False:
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, target_modules='all-linear')
        model = get_peft_model(model, peft_config)

    model = create_model()
    device = torch.device('cuda')
    model.to(device)
    #for param in model.model.encoder.parameters():
        #param.requires_grad = False
    data = preproc_data()
    data.to(device)
    print('about to train')
    outputs = model(data['input_ids'], data['tags'], attention_mask=data['attention_mask'], labels=data['labels'])
    #outputs = model(data['input_ids'], attention_mask=data['attention_mask'], labels=data['labels'])
    print(outputs)
    return
    dataloader = create_dataloaders()
    morph_custom_train(model, dataloader, dataloader, 1)
    return



    #set_default_device("cuda")
    #eval(model_checkpoint, 'se', 'en')
    #finetune_and_eval('jbochi/madlad400-3b-mt', 'MADLAD', 'en', 'fi')
    #finetune_and_eval('facebook/nllb-200-distilled-600M', 'NLLB', 'en', 'fi')
    finetune_and_eval('facebook/nllb-200-distilled-600M', 'Morph', 'en', 'fi')
    return

if __name__ == "__main__":
    main()
