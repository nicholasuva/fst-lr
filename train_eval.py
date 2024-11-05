#taking inspiration from this notebook
#https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb
#test test test
from transformers import PretrainedConfig, GenerationConfig, TrainerCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Config, M2M100Tokenizer, NllbTokenizer, get_scheduler, Adafactor
from datasets import Dataset, DatasetDict, load_from_disk, load_metric
from typing import Callable
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

from torch import cuda, Generator, no_grad, bfloat16, float16
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import AdamW
from evaluate import evaluator
import numpy as np
import utils
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
#from morph_tokenizer import create_morph_tokenizer

from morph_model import MorphM2M100, MorphModelDataCollator, DebugTrainer

import json
import argparse
import evaluate
import logging
from datetime import datetime, timedelta
import os
import re
import faulthandler

faulthandler.enable()
#import function_trace
#some stuff that will be used everywhere perhaps
#model_checkpoint = 'jbochi/madlad400-3b-mt'

#trying gpu version override
HSA_OVERRIDE_GFX_VERSION=1030

os.environ["TOKENIZERS_PARALLELISM"] = "false"


#torch.autograd.set_detect_anomaly(True)

#class GradientClippingCallback(TrainerCallback):
    #def on_step_end(self, args, state, control, **kwargs):
        #torch.nn.utils.clip_grad_norm_(self.model.parameters())


class GradientClippingCallback(TrainerCallback):
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Apply gradient clipping
        if model is not None:
            #print('clipping')
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

#in use
def compute_metrics(
        eval_preds
    ):
    """
    
    """
    #print(f"-----------compute metrics-----------")
    #print(f"eval preds fields: {[entry for entry in eval_preds]}")
    #print(f"eval preds content: {[eval_preds[entry] for entry in eval_preds]}")
    tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
    
    #tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    logits, labels = eval_preds
    logits_lang_codes = tokenizer.batch_decode([sent[0] for sent in logits], skip_special_tokens=True)
    labels_lang_codes = tokenizer.batch_decode([sent[0] for sent in labels], skip_special_tokens=True)
    #print(f"logit lang codes: {logits_lang_codes}")
    #print(f"label lang codes: {labels_lang_codes}")
    #print(f"logits {logits}")
    #print(f"labels: {labels}")
    if isinstance(logits, tuple):
        logits = logits[0]

    #preds = np.argmax(logits, axis=-1)
    preds = logits
    #labels = labels[0]
    #print(f"preds: {preds}")
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #print(f"decoded preds: {decoded_preds}")

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #print(f"labels: {labels}")
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #print(f"decoded labels: {decoded_labels}")

    # Some simple post-processing
    #decoded_preds = [pred.strip() for pred in preds]
    #decoded_labels = [[label.strip()] for label in labels]

    #load_metric is deprecated, need to use evaluate.load() from the library hf evaluate
    #bleu_metric = load_metric("sacrebleu", trust_remote_code=True)
    #accuracy_metric = load_metric("accuracy", trust_remote_code=True)

    #for i in range(len(decoded_labels)):
        #print(f"pred {i}: {decoded_preds[i]}")
        #print(f"label {i}: {decoded_labels[i]}")

    bleu_metric = evaluate.load('sacrebleu')
    #accuracy_metric = evaluate.load('accuracy')

    
    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
        )
    #accuracy = accuracy_metric.compute(
        #predictions=np.argmax(preds, axis=-1),
        #predictions=preds,
        #references=labels
        #)
    results = {
        'bleu': bleu['score']#,
        #'accuracy': accuracy['accuracy']
    }
    return results

#in use
def create_model():
    print('creating model')
    initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    morph_encoder_config = M2M100Config(
        vocab_size=1024,
        encoder_layers=2,
        d_model=1024,
        dropout=0.1,
        encoder_layerdrop=0,
        pad_token_id=1,
        max_position_embeddings=1024,
        scale_embedding=True,
    )
    #text_tokenizer = xxxxx
    #config = M2M100ForConditionalGeneration.from_pretrained(initial_checkpoint).config
    model = MorphM2M100(initial_checkpoint, morph_encoder_config)
    if False:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            #target_modules='all-linear'
            target_modules=['q_proj', 'v_proj']
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    print('model created')
    return model




#in use
def create_baseline_model():
    print('creating model')
    initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    #text_tokenizer = xxxxx
    #config = M2M100ForConditionalGeneration.from_pretrained(initial_checkpoint).config
    #model = MorphM2M100(config)
    model = M2M100ForConditionalGeneration.from_pretrained(initial_checkpoint)
    config_dict = model.config.to_dict()
    del config_dict['max_length']
    model.config = PretrainedConfig.from_dict(config_dict)
    model.generation_config.max_length=200
    #model = M2M100ForConditionalGeneration(config)
    if False:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            #target_modules='all-linear'
            target_modules=['q_proj', 'v_proj']
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    print('model created')
    return model

def create_morph_tokenizer(
        src_lang,
        trg_lang,     
):
    try:
        dataset = load_from_disk(f"{trg_lang}-{src_lang}-combined.hf")
    except:
        dataset = load_from_disk(f"{src_lang}-{trg_lang}-combined.hf")
    data = dataset['train'][f'{src_lang} tags']
    #print(data[0])
    data = [' '.join(tag for tag in tags) for tags in data]
    data = [sent.replace('+', '') for sent in data]
    data = [clean_text(sent) for sent in data]
    #print(data[0])
    tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
    tokenizer.train_from_iterator(data, trainer)
    tokenizer.save(f'{src_lang}_morph_tag_tokenizer.json')
    #test_sent = data[0]
    #encoded = tokenizer.encode(test_sent)
    #print(encoded.tokens)
    #print(encoded.ids)
    return tokenizer



def clean_text(text):
    text = re.sub(r"[^a-zA-Z\u0400-\u04FF0-9 .,!?'\-]", "", text)
    return text

#in use
def tokenize_dataset(
    text_tokenizer,
    tag_tokenizer,
    src_lang,
    trg_lang,
    max_length=128
    ):
    #load the dataset
    try:
        dataset_dict = load_from_disk(f"{src_lang}-{trg_lang}-combined.hf")
    except:
        dataset_dict = load_from_disk(f"{trg_lang}-{src_lang}-combined.hf")
    
    """
    tokenized_dict = {}
    for split in dataset_dict:
        cleaned_data = []
        #print(f"split: {split}")
        dataset = dataset_dict[split]
        inputs = [ex for ex in dataset[f"{src_lang}_text"]]
        targets = [ex for ex in dataset[f"{trg_lang}_text"]]
        tags_data = [ex for ex in dataset[f'{src_lang} tags']]
        tags_data = [' '.join(tag for tag in tags) for tags in tags_data]
        tags_data = [sent.replace('+', '') for sent in tags_data]
        model_inputs = text_tokenizer(inputs, text_target=targets, max_length=max_length, padding=False, truncation=True)
        model_tag_inputs = [tag_tokenizer.encode(tags).ids[:max_length] for tags in tags_data]
        model_inputs['tags'] = model_tag_inputs
        model_inputs = Dataset.from_dict(model_inputs)
        tokenized_dict[split] = model_inputs
    tokenized_dataset_dict = DatasetDict(tokenized_dict)
    #print(f"final processed thing: {tokenized_dataset_dict}")
    #return tokenized_dataset_dict
    """
    tokenized_dict = {}
    for split in dataset_dict:
        cleaned_data = []
        dataset = dataset_dict[split]
        for ex in dataset:
            inputs = ex[f"{src_lang}_text"]
            inputs = clean_text(inputs)
            targets = ex[f"{trg_lang}_text"]
            targets = clean_text(targets)
            tags = ex[f'{src_lang} tags']
            tags = ' '.join(tag for tag in tags)
            tags = tags.replace('+', '')
            tags = clean_text(tags)

            model_inputs = text_tokenizer(inputs, text_target = targets, max_length=max_length, padding=False, truncation=True)
            model_tag_inputs = tag_tokenizer.encode(tags).ids[:max_length]
            if len(model_inputs['input_ids'])>0 and len(model_inputs['labels'])>0 and len(model_tag_inputs)>0:
                cleaned_data.append(
                    {
                        'input_ids': model_inputs['input_ids'],
                        'labels': model_inputs['labels'],
                        'tags': model_tag_inputs
                    }
                )
        cleaned_data = Dataset.from_list(cleaned_data)
        tokenized_dict[split] = cleaned_data
    tokenized_dataset_dict = DatasetDict(tokenized_dict)
    return tokenized_dataset_dict




#deprecated
def gen_training_args(
        log_filepath,
        batch_size=1
):
    training_args = Seq2SeqTrainingArguments(
        #output_dir='./tt-en-init-test_results',
        output_dir=f"./{log_filepath}/training_results/",
        logging_dir=f"./{log_filepath}/training_logs/",
        fp16=True,
        eval_strategy='steps', #'no', 'steps', or 'epoch'
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        #learning_rate=5e-5,
        weight_decay=0.01,
        #logging_dir='./test_logs',
        logging_steps=500,
        save_steps=500,
        num_train_epochs=3,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model='eval_bleu',
        greater_is_better=True,
        save_total_limit=3,
        warmup_steps=10,

        #label_smoothing_factor=0.1
    )
    return training_args


def training_pipeline(
        initial_checkpoint,
        src_lang,
        trg_lang,
        load_from_disk=False,
        model_disk_filepath=None,
        save_model=True,
        save_to_cumulative=True,
        run_baseline=False,
        to_train=True,
        to_eval=True,
        proportion_of_train_dataset=1.0,
        proportion_of_test_dataset=1.0,
        mid_training_eval_sent_num=128,
        batch_size=1,
        expdir=''
):
    now = datetime.now() - timedelta(hours=5, minutes=0) #accounting for timezone
    now = now.strftime(f"%Y-%m-%d_%H-%M")
    model_mode = 'baseline' if run_baseline else 'experimental'
    all_logs_filepath = f"{expdir}/{src_lang}-{trg_lang}_mode-{model_mode}_train-{str(to_train)}_eval-{str(to_eval)}_{now}"
    #train_results_filepath = f"./{all_logs_filepath}/train_results"
    src_nllb = utils.get_nllb_code(src_lang)
    trg_nllb = utils.get_nllb_code(trg_lang)

    #CREATE TOKENIZERS
    text_tokenizer = NllbTokenizer.from_pretrained(
        initial_checkpoint,
        src_lang=src_nllb,
        tgt_lang=trg_nllb
        )
    tag_tokenizer = create_morph_tokenizer(src_lang=src_lang, trg_lang=trg_lang)
    
    #PREPROCESS DATA
    dataset_dict = tokenize_dataset(
        text_tokenizer=text_tokenizer,
        tag_tokenizer=tag_tokenizer,
        src_lang=src_lang,
        trg_lang=trg_lang
        )
    
    #CREATE MODEL AND DATA COLLATOR (need to add ability to load a model)
    if run_baseline:
        model = create_baseline_model()
        data_collator = DataCollatorForSeq2Seq(text_tokenizer, model)
    else:
        model = create_model()
        data_collator = MorphModelDataCollator(text_tokenizer, model)

    cumulative_model_path_holder_filename = "./cumulative_model.txt"
    trg_lang_nllb_code = utils.get_nllb_code(trg_lang)
    trg_lang_nllb_id = text_tokenizer.convert_tokens_to_ids(trg_lang_nllb_code)
    
    #LOAD OPTIMIZER
    optimizer = AdamW(
        model.parameters(),
        lr=3e-5
    )

    #PROPORTION OUT DATASETS
    if proportion_of_train_dataset>=1.0:
        train_dataset = dataset_dict['train']
    else:
        train_dataset = dataset_dict['train'].train_test_split(test_size=proportion_of_train_dataset)['test']
    mid_training_eval_proportion = mid_training_eval_sent_num / len(dataset_dict['test'])
    if mid_training_eval_proportion >= 1.0:
        mid_training_eval_set = dataset_dict['test']
    else:
        #mid_training_eval_proportion = 1.0
        mid_training_eval_set = dataset_dict['test'].train_test_split(test_size=mid_training_eval_proportion)['test']
    if proportion_of_test_dataset>=1.0:
        test_dataset = dataset_dict['test']
    else:
        test_dataset = dataset_dict['test'].train_test_split(test_size=proportion_of_test_dataset)['test']

    #SOME OPTIMIZATIONS
    #torch.autograd.set_detect_anomaly(True)
    torch.cuda.synchronize()
    #model.gradient_checkpointing_enable()

    #GENERATE TRAINING ARGUMENTS
    #generation_config = GenerationConfig(
        #max_length=200,
        #forced_bos_token_id=trg_lang_nllb_id
    #)

    training_args = Seq2SeqTrainingArguments(
        #output_dir='./tt-en-init-test_results',
        output_dir=f"./{all_logs_filepath}/training_results/",
        logging_dir=f"./{all_logs_filepath}/training_logs/",
        fp16=True,
        eval_strategy='steps', #'no', 'steps', or 'epoch'
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        #learning_rate=5e-5,
        weight_decay=0.01,
        #logging_dir='./test_logs',
        logging_steps=500,
        save_steps=500,
        num_train_epochs=3,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model='eval_bleu',
        greater_is_better=True,
        save_total_limit=3,
        warmup_steps=10,
        #label_smoothing_factor=0.1
        #generation_config=generation_config
    )


    #TRAINING
    if to_train:
        print('TRAIN')
        #training_args = gen_training_args(all_logs_filepath, batch_size=batch_size)
        training_trainer = Seq2SeqTrainer(
        #training_trainer = DebugTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=mid_training_eval_set,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
            callbacks=[GradientClippingCallback(max_norm=1.0)],
        )
        #does train return anything? I just added the output
        results = training_trainer.train()
        best_model_dir = training_trainer.state.best_model_checkpoint

        #TRAINING CLEANUP
        del training_trainer
        del train_dataset
        del mid_training_eval_set

    #EVALUATION
    if to_eval:
        print('EVAL')
        #eval_args = gen_training_args(all_logs_filepath, batch_size=batch_size)
        eval_trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            #train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
        )
        results = eval_trainer.evaluate(forced_bos_token_id=trg_lang_nllb_id)
        #results = trainer.evaluate()
        print(f"results: {results}")
        eval_results_filepath = f"./{all_logs_filepath}/evaluation_results.json"
        with open(eval_results_filepath, 'w') as sink:
            json.dump(results, sink, indent=2)
        #with open('fi-en-morph-embed-dim-cat-first_try_model_results.txt', 'a') as sink:
            #sink.write(str(results))
        best_model_dir = eval_trainer.state.best_model_checkpoint

        #EVAL CLEANUP
        del eval_trainer
        del test_dataset


    #this should only be the case if it's not baseline, will do once I combine these functions
    if save_model:
        if save_to_cumulative:
            #need to fix, why is this returning None?
            if best_model_dir is not None:
                with open(cumulative_model_path_holder_filename, 'w') as sink:
                    sink.write(best_model_dir)

    #OVERALL CLEANUP
    del model
    del optimizer
    del dataset_dict
    torch.cuda.empty_cache()

    return

def main(
        args: argparse.Namespace
) -> None:

    #CUDA ENABLED SANITY CHECK
    print(f"cuda device count: {cuda.device_count()}")
    print(f"cuda is available: {cuda.is_available()}")
    print(f"cuda current device: {cuda.current_device()}")
    print(f"cuda current device name: {cuda.get_device_name(cuda.current_device())}")

    #PARSE ARGUMENTS
    if(args.title):
        print(args.title)
    src_lang = args.src
    trg_lang = args.trg
    expdir = args.expdir
    to_train = True if args.train else False
    to_eval = True if args.eval else False
    run_baseline = True if args.baseline else False

    print(f"{src_lang}-{trg_lang}_train-{to_train}_eval-{to_eval}_is-baseline-{run_baseline}")


    #oct 28 test
    initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    run_langs = ['tt'] #just baseline plus experimental, not finetune

    training_pipeline(
        initial_checkpoint=initial_checkpoint,
        src_lang=src_lang,
        trg_lang=trg_lang,
        run_baseline=run_baseline,
        to_train=to_train,
        to_eval=to_eval,
        proportion_of_train_dataset=1.0,
        proportion_of_test_dataset=1.0,
        batch_size=8,
        expdir=expdir
    )
    return


    
    these_src = ['so', 'ba', 'fo', 'ga']
    this_trg = 'en'
    for this_src in these_src:
        print(this_src)
        """
        print('baseline')
        training_pipeline(
            initial_checkpoint=initial_checkpoint,
            src_lang=this_src,
            trg_lang=this_trg,
            run_baseline=True,
            to_train=False,
            to_eval=True,
            proportion_of_train_dataset=1.0,
            proportion_of_test_dataset=1.0
        )
        """
        print('Baseline')
        training_pipeline(
            initial_checkpoint=initial_checkpoint,
            src_lang=this_src,
            trg_lang=this_trg,
            run_baseline=True,
            to_train=False,
            to_eval=True,
            proportion_of_train_dataset=1.0,
            proportion_of_test_dataset=1.0,
            batch_size=8
        )
        print('Finetune')
        training_pipeline(
            initial_checkpoint=initial_checkpoint,
            src_lang=this_src,
            trg_lang=this_trg,
            run_baseline=True,
            to_train=True,
            to_eval=True,
            proportion_of_train_dataset=1.0,
            proportion_of_test_dataset=1.0,
            batch_size=8
        )
        print('experimental')
        training_pipeline(
            initial_checkpoint=initial_checkpoint,
            src_lang=this_src,
            trg_lang=this_trg,
            run_baseline=False,
            to_train=True,
            to_eval=True,
            proportion_of_train_dataset=1.0,
            proportion_of_test_dataset=1.0,
            batch_size=8
        )
    return
    print('finetune')
    training_pipeline(
        initial_checkpoint=initial_checkpoint,
        src_lang=this_src,
        trg_lang=this_trg,
        run_baseline=True,
        to_train=True,
        to_eval=True,
        proportion_of_train_dataset=1.0,
        proportion_of_test_dataset=1.0
    )
    print('experimental')
    training_pipeline(
        initial_checkpoint=initial_checkpoint,
        src_lang=this_src,
        trg_lang=this_trg,
        run_baseline=False,
        to_train=True,
        to_eval=True,
        proportion_of_train_dataset=1.0,
        proportion_of_test_dataset=1.0
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--title",
        required=False,
        type=str,
        help="a title to print to the console"
    )
    parser.add_argument(
        "--src",
        required=True,
        type=str,
        help="source language ISO 639 set 1 code"
    )
    parser.add_argument(
        "--trg",
        required=True,
        type=str,
        help="target language ISO 639 set 1 code"
    )
    parser.add_argument(
        "--expdir",
        required=True,
        type=str,
        help="directory for results of this experiment"
    )
    parser.add_argument(
        "--train",
        action='store_true',
        #required=True,
        #default=True,
        #type=bool,
        help="whether to train the model"
    )
    parser.add_argument(
        "--eval",
        action='store_true',
        #required=True,
        #default=True,
        #type=bool,
        help="whether to evaluate the model"
    )
    parser.add_argument(
        "--baseline",
        action='store_true',
        #required=True,
        #default=False,
        #type=bool,
        help="whether to run the baseline NLLB-200-distilled-600M model (instead of the experimental version)"
    )

    main(parser.parse_args())


"""
what args do I need
src lang
trg lang
to train
to eval
baseline or experimental
other exp parameters like idk, save and load cumulative, adjust lr, that kind of thing
"""