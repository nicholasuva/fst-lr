#taking inspiration from this notebook
#https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb
#test test test
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Config, M2M100Tokenizer, AdamW, NllbTokenizer, get_scheduler, Adafactor
from datasets import Dataset, DatasetDict, load_from_disk, load_metric
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

from morph_model import MorphM2M100, MorphModelDataCollator, ForwardOnlyTrainer

import evaluate


import sys
import trace
import psutil
import resource
import hunter
#import function_trace
#some stuff that will be used everywhere perhaps
#model_checkpoint = 'jbochi/madlad400-3b-mt'

#trying gpu version override
HSA_OVERRIDE_GFX_VERSION=1030




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
    print(f"-----------beginning compute metrics-----------")
    #print(f"eval preds fields: {[entry for entry in eval_preds]}")
    #print(f"eval preds content: {[eval_preds[entry] for entry in eval_preds]}")
    tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
    
    #tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    logits, labels = eval_preds
    logits_lang_codes = tokenizer.batch_decode([sent[0] for sent in logits], skip_special_tokens=True)
    labels_lang_codes = tokenizer.batch_decode([sent[0] for sent in labels], skip_special_tokens=True)
    print(f"logit lang codes: {logits_lang_codes}")
    print(f"label lang codes: {labels_lang_codes}")
    print(f"logits {logits}")
    print(f"labels: {labels}")
    if isinstance(logits, tuple):
        logits = logits[0]

    #preds = np.argmax(logits, axis=-1)
    preds = logits
    #labels = labels[0]
    #print(f"preds: {preds}")
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print(f"decoded preds: {decoded_preds}")

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #print(f"labels: {labels}")
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(f"decoded labels: {decoded_labels}")

    # Some simple post-processing
    #decoded_preds = [pred.strip() for pred in preds]
    #decoded_labels = [[label.strip()] for label in labels]

    #load_metric is deprecated, need to use evaluate.load() from the library hf evaluate
    #bleu_metric = load_metric("sacrebleu", trust_remote_code=True)
    #accuracy_metric = load_metric("accuracy", trust_remote_code=True)

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


"""
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
"""

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
            #gradient_accumulation_steps=4,
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
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            #target_modules='all-linear'
            target_modules=['q_proj', 'v_proj']
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    print('model created')
    return model

def create_baseline_model():
    print('creating model')
    initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    #text_tokenizer = xxxxx
    #config = M2M100ForConditionalGeneration.from_pretrained(initial_checkpoint).config
    #model = MorphM2M100(config)
    model = M2M100ForConditionalGeneration.from_pretrained(initial_checkpoint)
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

def preproc_data(text_tokenizer):
    src_lang = 'fi'
    trg_lang = 'en'
    tag_tokenizer = create_morph_tokenizer()
    max_length = 250
    max_input_length = max_length
    max_tag_length = max_length
    max_target_length = max_length
    dataset = load_from_disk('en-fi-combined.hf')
    #dataset= dataset['train'].select(range(1))
    dataset= dataset['train'].train_test_split(test_size=0.00005)['test']

    prefix = utils.get_nllb_code(trg_lang) + ' ' #this is wrong I need prefix on the trg lang for nllb ugh wait maybe this is right
    #inputs = [(prefix + ex[src_lang]) for ex in dataset['translation']]
    inputs = [(ex[src_lang]) for ex in dataset['translation']]
    targets = [(ex[trg_lang]) for ex in dataset['translation']]
    print(f"raw inputs: {[sent for sent in inputs]}")
    print(f"raw targets: {[sent for sent in targets]}")
    

    tags_data = [ex for ex in dataset['fi tags']]
    tags_data = [' '.join(tag for tag in tags) for tags in tags_data]
    tags_data = [sent.replace('+', '') for sent in tags_data]
    #print(tags_data[0])
    #print(tags_data[1])
    #trying out not padding here bc the data collator will pad
    #model_inputs = text_tokenizer(inputs, max_length=max_input_length, padding=True, truncation=True, return_tensors='pt')
    #labels = text_tokenizer(targets, max_length=max_target_length, padding=True, truncation=True, return_tensors='pt')
    print(f"inputs: {inputs}")
    print(f"targets: {targets}")
    print(f"max input length: {max_input_length}")
    model_inputs = text_tokenizer(inputs, text_target=targets, max_length=max_input_length, padding=False, truncation=True)
    #with text_tokenizer.as_target_tokenizer():
    #    labels = text_tokenizer(targets, max_length=max_target_length, padding=False, truncation=True)
    model_tag_inputs = [tag_tokenizer.encode(tags).ids[:max_tag_length] for tags in tags_data]
    

    #testing effect of snt ln on memory
    """
    data_trunc_test_len = 1000
    model_inputs['input_ids'] = [ex[:data_trunc_test_len] for ex in model_inputs['input_ids']]
    labels['input_ids'] = [ex[:data_trunc_test_len] for ex in labels['input_ids']]
    model_tag_inputs = [ex[:data_trunc_test_len] for ex in model_tag_inputs]
    #model_inputs['input_ids'] = [ex for ex in model_inputs['input_ids']]
    #labels['input_ids'] = [ex for ex in labels['input_ids']]
    #model_tag_inputs = [ex for ex in model_tag_inputs]
    """
    print(model_inputs)
    #model_tag_inputs = [enc + [0] * (max_input_length - len(enc)) if len(enc) < max_input_length else enc[:max_input_length] for enc in model_tag_inputs]
    #model_tag_inputs = torch.tensor(model_tag_inputs)
    #model_inputs['labels'] = labels['input_ids']
    model_inputs['tags'] = model_tag_inputs
    model_inputs = Dataset.from_dict(model_inputs)
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


def gen_training_args():
    training_args = Seq2SeqTrainingArguments(
        output_dir='./test_results',
        fp16=True,
        eval_strategy='steps',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        #learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir='./test_logs',
        logging_steps=100,
        save_steps=500,
        num_train_epochs=1,
        predict_with_generate=True,
        load_best_model_at_end=False,
        save_total_limit=3,
        warmup_steps=10,
        #label_smoothing_factor=0.1
    )
    return training_args


def morph_train_with_trainer(model, data, text_tokenizer, to_train=False, to_eval=False):
    src_lang = 'fi'
    trg_lang = 'en'
    #text_tokenizer = NllbTokenizer.from_pretrained(initial_checkpoint, src_lang=utils.get_nllb_code(src_lang))
    if isinstance(model, MorphM2M100):
        data_collator = MorphModelDataCollator(text_tokenizer, model)
    else:
        data_collator = DataCollatorForSeq2Seq(text_tokenizer, model)
    training_args = gen_training_args()
    #model.gradient_checkpointing_enable()
    optimizer = AdamW(
    #optimizer = Adafactor(
        model.parameters(),
        #scale_parameter=True,
        #relative_step=False,
        #warmup_init=False,
        lr=5e-5
    )

    trainer = Seq2SeqTrainer(
    #trainer = ForwardOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=data,
        eval_dataset=data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )
    if to_train:
        trainer.train()
    if to_eval:
        results = trainer.evaluate()
        print(f"results: {results}")
        with open('first_try_model_results.txt', 'w') as sink:
            sink.write(str(results))
    trainer.save_model('./first_try_test_model')
    
    return



def morph_forward_only_test():
    data = preproc_data()
    print(f"data type {type(data)}")
    #data=data[:1]
    print(f"data type {type(data)}")
    model = create_model()
    tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
    data_collator = MorphModelDataCollator(tokenizer, model)
    data = data_collator(data)
    forward_outputs = model.forward(input_ids=data['input_ids'], tags=data['tags'], labels=data['labels'], attention_mask=data['attention_mask'])
    return forward_outputs


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
    #cuda.empty_cache()
    #cuda.reset_max_memory_allocated()
    #print(cuda.memory_summary())
    #finetune_and_eval('facebook/nllb-200-distilled-600M', 'Morph', 'en', 'fi')
    #return


    #model = M2M100ForConditionalGeneration.from_pretrained('facebook/nllb-200-distilled-600m')

    model = create_model()

    #model = create_baseline_model()
    #model.config.decoder_start_token_id = 256047
    #model.config.bos_token_id = 256042
    #model.gradient_checkpointing_enable()
    src_lang = 'fi'
    trg_lang = 'en'
    src_nllb = utils.get_nllb_code(src_lang)
    trg_nllb = utils.get_nllb_code(trg_lang)
    print(src_nllb)
    print(trg_nllb)
    initial_checkpoint = 'facebook/nllb-200-distilled-600M'
    text_tokenizer = NllbTokenizer.from_pretrained(
        initial_checkpoint,
        src_lang=src_nllb,
        tgt_lang=trg_nllb
        )

    #device = torch.device('cuda')
    #model.to(device)
    #for param in model.model.encoder.parameters():
        #param.requires_grad = False
    data = preproc_data(text_tokenizer)
    #data.to(device)
    print('about to train')
    morph_train_with_trainer(model, data, text_tokenizer, to_train=False, to_eval=True)
    print(f"dataset size: {data.num_rows}")
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
