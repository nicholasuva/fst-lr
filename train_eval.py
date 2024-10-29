#taking inspiration from this notebook
#https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb
#test test test
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Config, M2M100Tokenizer, AdamW, NllbTokenizer, get_scheduler, Adafactor
from datasets import Dataset, DatasetDict, load_from_disk, load_metric
from typing import Callable
from torch import cuda, Generator, no_grad, bfloat16, float16
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from evaluate import evaluator
import numpy as np
import utils
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
from morph_tokenizer import create_morph_tokenizer

from morph_model import MorphM2M100, MorphModelDataCollator

import evaluate
import logging
from datetime import datetime

#import function_trace
#some stuff that will be used everywhere perhaps
#model_checkpoint = 'jbochi/madlad400-3b-mt'

#trying gpu version override
HSA_OVERRIDE_GFX_VERSION=1030


#in use
def compute_metrics(
        eval_preds
    ):
    """
    
    """
    print(f"-----------compute metrics-----------")
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

#in use
def tokenize_dataset(
    text_tokenizer,
    tag_tokenizer,
    src_lang,
    trg_lang,
    max_length=200
    ):
    #load the dataset
    try:
        dataset_dict = load_from_disk(f"{src_lang}-{trg_lang}-combined.hf")
    except:
        dataset_dict = load_from_disk(f"{trg_lang}-{src_lang}-combined.hf")
    tokenized_dict = {}
    for split in dataset_dict:
        print(f"split: {split}")
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
    print(f"final processed thing: {tokenized_dataset_dict}")
    return tokenized_dataset_dict

#in use
def gen_training_args(
        log_filepath
):
    training_args = Seq2SeqTrainingArguments(
        #output_dir='./tt-en-init-test_results',
        output_dir=f"./{log_filepath}/training_results/",
        logging_dir=f"./{log_filepath}/training_logs/",
        fp16=True,
        eval_strategy='epoch', #'no', 'steps', or 'epoch'
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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

#in use
def morph_train_with_trainer(
        model,
        data,
        text_tokenizer,
        src_lang,
        trg_lang,
        to_train=False,
        to_eval=False,
        save_model=False,
        save_to_cumulative=False,
        proportion_of_train_dataset=1.0,
        proportion_of_test_dataset=1.0,
        mid_training_eval_sent_num=100
        ):
    cumulative_model_path_holder_filename = "./cumulative_model.txt"
    trg_lang_nllb_code = utils.get_nllb_code(trg_lang)
    trg_lang_nllb_id = text_tokenizer.convert_tokens_to_ids(trg_lang_nllb_code)
    #text_tokenizer = NllbTokenizer.from_pretrained(initial_checkpoint, src_lang=utils.get_nllb_code(src_lang))
    #if False:
    if isinstance(model, MorphM2M100):
        data_collator = MorphModelDataCollator(text_tokenizer, model)
    else:
        data_collator = DataCollatorForSeq2Seq(text_tokenizer, model)
    
    #model.gradient_checkpointing_enable()
    optimizer = AdamW(
    #optimizer = Adafactor(
        model.parameters(),
        #scale_parameter=True,
        #relative_step=False,
        #warmup_init=False,
        lr=5e-5
    )

    if proportion_of_train_dataset>=1.0:
        train_dataset = data['train']
    else:
        train_dataset = data['train'].train_test_split(test_size=proportion_of_train_dataset)['test']
    mid_training_eval_proportion = mid_training_eval_sent_num / len(data['test'])
    if mid_training_eval_proportion > 1.0:
        mid_training_eval_proportion = 1.0
    mid_training_eval_set = data['test'].train_test_split(test_size=mid_training_eval_proportion)['test']
    if proportion_of_test_dataset>=1.0:
        test_dataset = data['test']
    else:
        test_dataset = data['test'].train_test_split(test_size=proportion_of_test_dataset)['test']

    #TRAINING
    if to_train:
        training_args = gen_training_args()
        training_trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=mid_training_eval_set,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
        )
        #does train return anything? I just added the output
        results = training_trainer.train()
        best_model_dir = training_trainer.state.best_model_checkpoint
        

    #EVALUATION
    if to_eval:
        eval_args = gen_training_args()
        eval_trainer = Seq2SeqTrainer(
            model=model,
            args=eval_args,
            #train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
        )
        results = eval_trainer.evaluate(forced_bos_token_id=trg_lang_nllb_id)
        #results = trainer.evaluate()
        print(f"results: {results}")
        #with open('fi-en-morph-embed-dim-cat-first_try_model_results.txt', 'a') as sink:
            #sink.write(str(results))
        best_model_dir = eval_trainer.state.best_model_checkpoint


    if save_model:
        if save_to_cumulative:
            with open(cumulative_model_path_holder_filename, 'w') as sink:
                sink.write(best_model_dir)

    #trainer.save_model('./fi-en-morph-embed-dim-cat-first_try_test_model')
    
    return

def training_pipeline(
        initial_checkpoint,
        src_lang,
        trg_lang,
        load_from_disk=False,
        model_disk_filepath=None,
        save_model=False,
        run_baseline=False,
        to_train=True,
        to_eval=True,
        proportion_of_train_dataset=1.0,
        proportion_of_test_dataset=1.0
):
    
    all_logs_filepath = f"{src_lang}-{trg_lang}_train-{str(to_train)}_eval-{to_eval}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}"
    train_results_filepath = f"./{all_logs_filepath}/train_results"
    src_nllb = utils.get_nllb_code(src_lang)
    trg_nllb = utils.get_nllb_code(trg_lang)
    text_tokenizer = NllbTokenizer.from_pretrained(
        initial_checkpoint,
        src_lang=src_nllb,
        tgt_lang=trg_nllb
        )
    tag_tokenizer = create_morph_tokenizer()
    dataset_dict = tokenize_dataset(
        text_tokenizer=text_tokenizer,
        tag_tokenizer=tag_tokenizer,
        src_lang=src_lang,
        trg_lang=trg_lang
        )
    if run_baseline:
        model = create_baseline_model()
    else:
        model = create_model()



    morph_train_with_trainer(
        model,
        dataset_dict,
        text_tokenizer,
        src_lang=src_lang,
        trg_lang=trg_lang,
        to_train=to_train,
        to_eval=to_eval,
        proportion_of_train_dataset=proportion_of_train_dataset,
        proportion_of_test_dataset=proportion_of_test_dataset
    )
    

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

    print(f"cuda device count: {cuda.device_count()}")
    print(f"cuda is available: {cuda.is_available()}")
    print(f"cuda current device: {cuda.current_device()}")
    print(f"cuda current device name: {cuda.get_device_name(cuda.current_device())}")

    #oct 28 test
    initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    training_pipeline(
        initial_checkpoint=initial_checkpoint,
        src_lang='tt',
        trg_lang='en',
        run_baseline=False,
        to_train=True,
        to_eval=True,
        proportion_of_train_dataset=0.1,
        proportion_of_test_dataset=0.1
    )
    return

    #cuda.empty_cache()
    #cuda.reset_max_memory_allocated()
    #print(cuda.memory_summary())
    #finetune_and_eval('facebook/nllb-200-distilled-600M', 'Morph', 'en', 'fi')
    #return


    #model = M2M100ForConditionalGeneration.from_pretrained('facebook/nllb-200-distilled-600m')

    src_lang = 'fi'
    trg_lang = 'en'
    src_nllb = utils.get_nllb_code(src_lang)
    trg_nllb = utils.get_nllb_code(trg_lang)
    initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    text_tokenizer = NllbTokenizer.from_pretrained(
        initial_checkpoint,
        src_lang=src_nllb,
        tgt_lang=trg_nllb
        )
    tag_tokenizer = create_morph_tokenizer()
    dataset_dict = tokenize_dataset(
        text_tokenizer=text_tokenizer,
        tag_tokenizer=tag_tokenizer,
        src_lang=src_lang,
        trg_lang=trg_lang
        )



    #################################
    model = create_model()
    #model = create_baseline_model()
    ##############################

    #model.config.decoder_start_token_id = 256047
    #model.config.bos_token_id = 256042
    #model.gradient_checkpointing_enable()
    

    #device = torch.device('cuda')
    #model.to(device)
    #for param in model.model.encoder.parameters():
        #param.requires_grad = False
    #data = preproc_data(text_tokenizer)
    #data.to(device)
    print('about to train')
    morph_train_with_trainer(model, dataset_dict, text_tokenizer, to_train=True, to_eval=True)
    #print(f"dataset size: {data.num_rows}")
    #outputs = model(data['input_ids'], data['tags'], attention_mask=data['attention_mask'], labels=data['labels'])
    #outputs = model(data['input_ids'], attention_mask=data['attention_mask'], labels=data['labels'])
    #print(outputs)
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
