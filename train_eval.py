#taking inspiration from this notebook
#https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

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
from morph_model import MorphM2M100, MorphModelDataCollator
import gc
import tracemalloc
import shutil
import json
import argparse
import evaluate
import logging
#from datetime import datetime, timedelta
import os
import re
import faulthandler
import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

faulthandler.enable()
tracemalloc.start()
#gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_LEAK)
#import function_trace


#trying gpu version override
HSA_OVERRIDE_GFX_VERSION=1030

os.environ["TOKENIZERS_PARALLELISM"] = "false"




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
def create_MorphM2M_model(
        initial_checkpoint,
        morph_encoder_layers=2,
        morph_d_model=512,
        morph_dropout=0.2,
):
    """
    initialize a MorphM2M model by loading a M2M100 model and modifying it to add a second encoder
    """
    print(f'creating MorphM2M model from initial checkpoint: {initial_checkpoint}')
    #initial_checkpoint = 'facebook/nllb-200-distilled-600M'
   
    #text_tokenizer = xxxxx
    #config = M2M100ForConditionalGeneration.from_pretrained(initial_checkpoint).config
    model = MorphM2M100(
        initial_checkpoint,
        morph_encoder_layers=morph_encoder_layers,
        morph_d_model=morph_d_model,
        morph_dropout=morph_dropout,
    )
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

def load_MorphM2M_model(
    initial_checkpoint
):
    """
    load a MorphM2M dual encoder model from a local checkpoint
    """
    #can I just inherit from_pretrained? let's see
    model = MorphM2M100.from_pretrained(initial_checkpoint)
    return model


#in use
def load_M2M100_model(
        initial_checkpoint
):
    """
    load a M2M100 model from a local or hub checkpoint

    """
    print(f'loading M2M100 model from checkpoint: {initial_checkpoint}')
    #initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    #text_tokenizer = xxxxx
    #config = M2M100ForConditionalGeneration.from_pretrained(initial_checkpoint).config
    #model = MorphM2M100(config)
    model = M2M100ForConditionalGeneration.from_pretrained(initial_checkpoint)
    config_dict = model.config.to_dict()
    if 'max_length' in config_dict:
        del config_dict['max_length']
        model.config = M2M100Config.from_dict(config_dict)
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
    print(f'model created, model type: {type(model)}')
    return model

def create_morph_tokenizer(
        src_lang,
        trg_lang,    
        expdir 
):
    try:
        dataset = load_from_disk(f"{trg_lang}-{src_lang}-combined.hf")
    except:
        dataset = load_from_disk(f"{src_lang}-{trg_lang}-combined.hf")
    data = dataset['train'][f'{src_lang} tags']
    #print(data[0])
    data = [clean_tags(tags) for tags in data]
    #data = [' '.join(tag for tag in tags) for tags in data]
    #data = [sent.replace('+', '') for sent in data]
    #data = [clean_text(sent) for sent in data]
    #print(data[0])
    tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[CLS]', '[SEP]'])
    tokenizer.train_from_iterator(data, trainer)
    tokenizers_path = f'./{expdir}/tokenizers'
    if not os.path.exists(tokenizers_path):
        os.makedirs(tokenizers_path)
    tokenizer.save(f'{tokenizers_path}/{src_lang}_morph_tag_tokenizer.json')
    #test_sent = data[0]
    #encoded = tokenizer.encode(test_sent)
    #print(encoded.tokens)
    #print(encoded.ids)
    return tokenizer

def clean_tags(
        tag_seq
        ):
    """
    clean and preprocess a single sequence of tags (corresponding to one sentence)
    """
    tags = ' '.join(tag for tag in tag_seq)
    #print(f"tags raw: {tags}")
    #ONE TOKEN PER TAG VS. ONE TOKEN PER FEATURE
    #uncomment this line for one token per tag (exp1, exp2)
    #tags = tags.replace('+', '')
    #uncomment this line for one token per feature (exp3, )
    tags = tags.replace('+', ' ')

    tags = clean_text(tags)
    #print(f"tags cleaned: {tags}")
    return tags


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
            tags = clean_tags(tags)
            model_inputs = text_tokenizer(inputs, text_target = targets, max_length=max_length, padding=False, truncation=True)
            model_tag_inputs = tag_tokenizer.encode(tags).ids[:max_length]
            #print(f" len of tags tokenized: {len(model_tag_inputs)}")
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

def choose_highest_checkpoint(
        parent_dir_path
):
    """
    takes in a parent, parses the names of the child dirs, returns the one with the highest numbered checkpoint if present
    """
    subdirs = []
    max_num = 0
    highest_dir = ''
    for entry in os.scandir(parent_dir_path):
        if entry.is_dir():
            #print(entry)
            #print(entry.name)
            subdirs.append(entry.name)
    for dir in subdirs:
        result = re.search(r"checkpoint-(\d+)", dir)
        #print(result)
        #print(result.group(1))
        train_num = result.group(1)
        if train_num is not None:
            if int(train_num) > max_num:
                max_num = int(train_num)
                highest_dir = dir
    if highest_dir != '':
        res_path = f"{parent_dir_path}/{highest_dir}"
        return res_path
    else:
        print('no checkpoint directories found')
        return


def training_pipeline(
        initial_checkpoint,
        hub_checkpoint,
        src_lang,
        trg_lang,
        save_model=True,
        save_to_cumulative=True,
        run_baseline=False,
        load_morph=False,
        create_morph=False,
        to_train=True,
        to_eval=True,
        proportion_of_train_dataset=1.0,
        proportion_of_test_dataset=1.0,
        mid_training_eval_sent_num=128,
        batch_size=1,
        learning_rate=3e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        morph_dropout=None,
        morph_encoder_layers=None,
        morph_d_model=None,
        expdir='',
        save_strategy='epoch',
        evaluation_strategy='epoch',
        is_wandb_sweep=False,
        report_to='tensorboard'  
):
    #INITIAL FILEPATH VALUES
    model_mode = 'baseline' if run_baseline else 'experimental'
    cumulative_model_path_holder_filename = "./cumulative_model.txt"
    all_logs_filepath = f"./{expdir}/{src_lang}-{trg_lang}_mode-{model_mode}_train-{str(to_train)}_eval-{str(to_eval)}"
    if not os.path.exists(all_logs_filepath):
        os.makedirs(all_logs_filepath)

    #CREATE TOKENIZERS
    src_nllb = utils.get_nllb_code(src_lang)
    trg_nllb = utils.get_nllb_code(trg_lang)
    text_tokenizer = NllbTokenizer.from_pretrained(
        hub_checkpoint,
        src_lang=src_nllb,
        tgt_lang=trg_nllb
    )
    trg_lang_nllb_id = text_tokenizer.convert_tokens_to_ids(trg_nllb)
    tag_tokenizer = create_morph_tokenizer(
        src_lang=src_lang,
        trg_lang=trg_lang,
        expdir=expdir
    )
    

    #PREPROCESS DATA
    dataset_dict = tokenize_dataset(
        text_tokenizer=text_tokenizer,
        tag_tokenizer=tag_tokenizer,
        src_lang=src_lang,
        trg_lang=trg_lang
        )
    

    #CREATE MODEL AND DATA COLLATOR
    if run_baseline:
        model = load_M2M100_model(initial_checkpoint)
        data_collator = DataCollatorForSeq2Seq(text_tokenizer, model)
    else:
        if create_morph:
            if morph_dropout is not None and morph_encoder_layers is not None and morph_d_model is not None:
                model= create_MorphM2M_model(
                    initial_checkpoint,
                    morph_dropout=morph_dropout,
                    morph_encoder_layers=morph_encoder_layers,
                    morph_d_model=morph_d_model
                )
                assert model.morph_encoder.config.dropout == morph_dropout
                assert model.morph_encoder.config.encoder_layers == morph_encoder_layers
                assert model.morph_encoder.config.d_model == morph_d_model
            else:
                model = create_MorphM2M_model(initial_checkpoint)
        elif load_morph:
            model = load_MorphM2M_model(initial_checkpoint)
        data_collator = MorphModelDataCollator(text_tokenizer, model)
    

    #LOAD OPTIMIZER
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate
    )


    #PROPORTION OUT DATASETS
    if is_wandb_sweep:
        train_dataset=dataset_dict['dev']
        if len(train_dataset) > 500:
            train_dataset = train_dataset.select(range(500))
    else:
        train_dataset=dataset_dict['train']

    if proportion_of_train_dataset<1.0:
        train_dataset = train_dataset.train_test_split(test_size=proportion_of_train_dataset)['test']
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
    torch.cuda.synchronize()
    #torch.autograd.set_detect_anomaly(True)
    #model.gradient_checkpointing_enable()


    #CREATE TRAINING ARGUMENTS
    if is_wandb_sweep:
        output_dir = f"./wandb_sweep_tmp/"
        load_best_model_at_end=False
    else:
        output_dir = f"./{all_logs_filepath}/training_results/"
        load_best_model_at_end=True
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        logging_dir=f"./{all_logs_filepath}/training_logs/",
        fp16=True,
        eval_strategy=evaluation_strategy, #'no', 'steps', or 'epoch'
        save_strategy=save_strategy,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        logging_steps=500,
        save_steps=500,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model='eval_bleu',
        greater_is_better=True,
        save_total_limit=1,
        warmup_steps=10,
        report_to=report_to
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
        #best_model_dir = training_trainer.state.best_model_checkpoint

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
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
        )
        with torch.no_grad():
            results = eval_trainer.evaluate(forced_bos_token_id=trg_lang_nllb_id)
        print(f"results: {results}")
        eval_results_filepath = f"./{all_logs_filepath}/evaluation_results.json"
        with open(eval_results_filepath, 'w') as sink:
            json.dump(results, sink, indent=2)
        #best_model_dir = eval_trainer.state.best_model_checkpoint

        #EVAL CLEANUP
        del eval_trainer
        del test_dataset


    #this should only be the case if it's not baseline, will do once I combine these functions
    #if save_model:
        #if save_to_cumulative:
            #need to fix, why is this returning None?
            #if best_model_dir is not None:
                #with open(cumulative_model_path_holder_filename, 'w') as sink:
                    #sink.write(best_model_dir)

    #OVERALL CLEANUP
    del model
    del optimizer
    del dataset_dict
    del data_collator
    del text_tokenizer
    del tag_tokenizer
    gc.collect()
    print(f"gc.garbage: {gc.garbage}")
    torch.cuda.empty_cache()
    if is_wandb_sweep and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if results is not None:
        return results
    else:
        return

#DEPRECATED
def wandb_sweep(
    initial_checkpoint,
    hub_checkpoint,
    src_lang,
    trg_lang,
    lr_min=1e-5,
    lr_max=1e-3,
    batch_sizes=[4, 8, 16],
    epochs_min=3,
    epochs_max=10,
    n_trials=20,
    expdir=""
):
    wandb.init(project=f"{src_lang}-{trg_lang}-hyperparameter_tuning")
    def hyperparameter_search(trial):
        """
        this is going to be roughly equivalent to training_pipeline but for hyperparam tuning
        """
        learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", batch_sizes)
        num_train_epochs = trial.suggest_int("num_train_epochs", epochs_min, epochs_max)
        print(f"learning rate: {learning_rate}\tbatch size: {batch_size}\tnum train epochs: {num_train_epochs}")
        #copy training args from uhhhhhh pipeline?
        #wait a second maybe I can actually just call training pipeline looool
        try:
            snapshot_before_train = tracemalloc.take_snapshot()
            print("before train")
            #print(f" gc objects: {gc.get_objects()}")
            #for stat in snapshot_before_train[:10]:
                #print(stat)
            results = training_pipeline(
                initial_checkpoint,
                hub_checkpoint,
                src_lang,
                trg_lang,
                save_model=False,
                save_to_cumulative=False,
                run_baseline=False,
                load_morph=False,
                create_morph=True,
                to_train=True,
                to_eval=True,
                proportion_of_train_dataset=1.0,
                proportion_of_test_dataset=1.0,
                mid_training_eval_sent_num=128,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                expdir=expdir,
                save_strategy='no',
                evaluation_strategy='no',
                is_wandb_sweep=True,
                report_to='wandb'
            )
            snapshot_after_train = tracemalloc.take_snapshot()
            print("after train")
            #print(f" gc objects: {gc.get_objects()}")
            #for stat in snapshot_after_train[:10]:
                #print(stat)
            #top_diffs = snapshot_after_train.compare_to(snapshot_before_train,'lineno')
            #print("memory differences before and after training pipeline:")
            #for stat in top_diffs[:10]:
                #print(stat)
            wandb.log({
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_train_epochs": num_train_epochs,
                "eval_bleu": results['eval_bleu']
                })
            #then basically I want to do what pipeline does, train, eval, then get the bleu out of the eval and return the negative of the bleu,
            #because the optuna minimizes by default
            return (-1 * results['eval_bleu'])
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                wandb.log({
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_train_epochs": num_train_epochs,
                "Out of Memory Error": True
                })
                return float("inf")
            else:
                raise
    wandb_callback = WeightsAndBiasesCallback()
    study = optuna.create_study(direction="minimize")
    study.optimize(hyperparameter_search, n_trials=n_trials, callbacks=[wandb_callback])
    wandb.log(study.best_params)
    return

def main(
        args: argparse.Namespace
) -> None:

    #CUDA ENABLED SANITY CHECK
    print(f"cuda device count: {cuda.device_count()}")
    print(f"cuda is available: {cuda.is_available()}")
    print(f"cuda current device: {cuda.current_device()}")
    print(f"cuda current device name: {cuda.get_device_name(cuda.current_device())}")


    #INITIAL FILEPATH VALUES
    hub_checkpoint = 'facebook/nllb-200-distilled-600M'


    #PARSE ARGUMENTS
    if(args.title):
        print(args.title)
    src_lang = args.src
    trg_lang = args.trg
    expdir = args.expdir
    if not os.path.exists(f"./{expdir}"):
        os.makedirs(f"./{expdir}")
    to_train = True if args.train else False
    to_eval = True if args.eval else False
    run_baseline = True if args.baseline else False
    create_morph = True if args.createmorph else False
    load_morph = True if args.loadmorph else False
    is_wandb_sweep = True if args.wandbsweep else False
    if(args.initchkpt):
        initial_checkpoint = args.initchkpt
        if args.choosehighchkpt:
            initial_checkpoint = choose_highest_checkpoint(initial_checkpoint)
    else:
        initial_checkpoint = hub_checkpoint
    if(args.learningrate):
        learning_rate = args.learningrate
    if(args.batchsize):
        batch_size = args.batchsize
    if(args.numtrainepochs):
        num_train_epochs = args.numtrainepochs
    if(args.weightdecay):
        weight_decay = args.weightdecay
    if(args.dropout):
        morph_dropout = args.dropout
    if(args.encoderlayers):
        morph_encoder_layers = args.encoderlayers
    if(args.dmodel):
        morph_d_model = args.dmodel

    
    if is_wandb_sweep:
    #RUN A SINGLE TEST FROM THE WANDB HYPERPARAMETER SEARCH SWEEP
        #EXPERIMENTAL MODEL SWEEP
        if create_morph:
            results = training_pipeline(
                initial_checkpoint=initial_checkpoint,
                hub_checkpoint=hub_checkpoint,
                src_lang=src_lang,
                trg_lang=trg_lang,
                save_model=False,
                save_to_cumulative=False,
                run_baseline=False,
                load_morph=False,
                create_morph=True,
                to_train=True,
                to_eval=True,
                proportion_of_train_dataset=1.0,
                proportion_of_test_dataset=1.0,
                mid_training_eval_sent_num=128,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                weight_decay=weight_decay,
                morph_dropout=morph_dropout,
                morph_encoder_layers=morph_encoder_layers,
                morph_d_model=morph_d_model,
                expdir=expdir,
                save_strategy='no',
                evaluation_strategy='no',
                is_wandb_sweep=True,
                report_to='wandb'
            )
        #BASELINE FINETUNE SWEEP
        elif run_baseline:
            results = training_pipeline(
                initial_checkpoint=initial_checkpoint,
                hub_checkpoint=hub_checkpoint,
                src_lang=src_lang,
                trg_lang=trg_lang,
                save_model=False,
                save_to_cumulative=False,
                run_baseline=True,
                load_morph=False,
                create_morph=False,
                to_train=True,
                to_eval=True,
                proportion_of_train_dataset=1.0,
                proportion_of_test_dataset=1.0,
                mid_training_eval_sent_num=128,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                weight_decay=weight_decay,
                expdir=expdir,
                save_strategy='no',
                evaluation_strategy='no',
                is_wandb_sweep=True,
                report_to='wandb'
            )
        print(f"EVAL_BLEU_RESULT: {results['eval_bleu']}")
    else:
        #RUN TRAINING AND/OR EVALUATION
        print(f"{src_lang}-{trg_lang}_train-{to_train}_eval-{to_eval}_is-baseline-{run_baseline}")
        training_pipeline(
            initial_checkpoint=initial_checkpoint,
            hub_checkpoint=hub_checkpoint,
            src_lang=src_lang,
            trg_lang=trg_lang,
            run_baseline=run_baseline,
            create_morph=create_morph,
            load_morph=load_morph,
            to_train=to_train,
            to_eval=to_eval,
            proportion_of_train_dataset=1.0,
            proportion_of_test_dataset=1.0,
            batch_size=8,
            expdir=expdir,
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
        "--initchkpt",
        required=False,
        type=str,
        help="directory for the model to train off of, if none, will default to the nllb model on the hub"
    )
    parser.add_argument(
        "--train",
        action='store_true',
        help="whether to train the model"
    )
    parser.add_argument(
        "--eval",
        action='store_true',
        help="whether to evaluate the model"
    )
    parser.add_argument(
        "--baseline",
        action='store_true',
        help="whether to run the baseline NLLB-200-distilled-600M model (instead of the experimental version)"
    )
    parser.add_argument(
        "--choosehighchkpt",
        action='store_true',
        help="whether to choose the h"
    )
    parser.add_argument(
        "--createmorph",
        action='store_true',
        help="whether to create a morphm2m model from an unmodified model"
    )
    parser.add_argument(
        "--loadmorph",
        action='store_true',
        help="whether to create a morphm2m model from an unmodified model"
    )
    parser.add_argument(
        "--wandbsweep",
        action='store_true',
        help="whether to perform a wandb sweep for the language pair"
    )
    parser.add_argument(
        "--learningrate",
        type=float,
        #action='store_true',
        help="learning rate for training"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        #action='store_true',
        help="batch size for training"
    )
    parser.add_argument(
        "--numtrainepochs",
        type=int,
        #action='store_true',
        help="number of epochs for training"
    )
    parser.add_argument(
        "--weightdecay",
        type=float,
        #action='store_true',
        help="weight decay for training"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        #action='store_true',
        help="dropout for morph encoder"
    )
    parser.add_argument(
        "--encoderlayers",
        type=int,
        #action='store_true',
        help="num layers for morph encoder"
    )
    parser.add_argument(
        "--dmodel",
        type=int,
        #action='store_true',
        help="model dimension for morph encoder"
    )

    main(parser.parse_args())


