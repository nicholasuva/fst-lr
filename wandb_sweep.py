
import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import subprocess
import re
import sys
#import train_eval


def wandb_sweep(
    initial_checkpoint,
    src_lang,
    trg_lang,
    is_baseline,
    expdir="",
    n_trials=20,
    lr_min=5e-6,
    lr_max=5e-4,
    batch_sizes=[4, 8, 16],
    epochs_min=3,
    epochs_max=15,
    weight_decay_min=0.01,
    weight_decay_max=0.1,
    dropout_min=0.1,
    dropout_max=0.5,
    encoder_layers_min=1,
    encoder_layers_max=8,
    d_model_range=[256, 512, 768, 1024]
):
    def hyperparameter_search(trial):
        """
        this is going to be roughly equivalent to training_pipeline but for hyperparam tuning
        """
        #TRAINING ARGS
        learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", batch_sizes)
        num_train_epochs = trial.suggest_int("num_train_epochs", epochs_min, epochs_max)
        weight_decay = trial.suggest_float("weight_decay", weight_decay_min, weight_decay_max)
        
        #MORPH ENCODER ARCHITECTURE
        if is_baseline:
            dropout = None
            encoder_layers = None
            d_model = None
        else:
            dropout = trial.suggest_float("dropout", dropout_min, dropout_max)
            encoder_layers = trial.suggest_int("encoder_layers", encoder_layers_min, encoder_layers_max)
            d_model = trial.suggest_categorical("d_model", d_model_range)
        
        #PRINT CURRENT TRIAL VALUES
        print(f"src_lang: {src_lang}", end='\t')
        print(f"learning rate: {learning_rate}", end='\t')
        print(f"batch size: {batch_size}", end='\t')
        print(f"num train epochs: {num_train_epochs}", end='\t')
        print(f"weight decay: {weight_decay}", end='\t')
        print(f"dropout: {dropout}", end='\t')
        print(f"encoder layers: {encoder_layers}", end='\t')
        print(f"d model: {d_model}", end='\t')
        print("")

        #SET FILEPATHS



        #CREATE BASELINE SUBPROCESS COMMAND
        if is_baseline:
            command = [
                    "python3", 
                    "train_eval.py",
                    f"--title", f"{src_lang}_baseline_wandb_sweep",
                    f"--src", f"{src_lang}",
                    f"--trg", f"{trg_lang}",
                    "--baseline",
                    "--wandbsweep",
                    "--train",
                    "--eval",
                    f"--expdir", f"{expdir}",
                    f"--initchkpt", f"{initial_checkpoint}",
                    f"--learningrate", f"{learning_rate}",
                    f"--batchsize", f"{batch_size}",
                    f"--numtrainepochs", f"{num_train_epochs}",
                    f"--weightdecay", f"{weight_decay}",
            ]
        #CREATE EXPERIMENTAL SUBPROCESS COMMAND
        else:
            command = [
                    "python3", 
                    "train_eval.py",
                    f"--title", f"{src_lang}_wandb_sweep",
                    f"--src", f"{src_lang}",
                    f"--trg", f"{trg_lang}",
                    "--createmorph",
                    "--wandbsweep",
                    "--train",
                    "--eval",
                    f"--expdir", f"{expdir}",
                    f"--initchkpt", f"{initial_checkpoint}",
                    "--choosehighchkpt",
                    f"--learningrate", f"{learning_rate}",
                    f"--batchsize", f"{batch_size}",
                    f"--numtrainepochs", f"{num_train_epochs}",
                    f"--weightdecay", f"{weight_decay}",
                    f"--dropout", f"{dropout}",
                    f"--encoderlayers", f"{encoder_layers}",
                    f"--dmodel", f"{d_model}",
            ]
        
        #copy training args from uhhhhhh pipeline?
        #wait a second maybe I can actually just call training pipeline looool
        try:

            #print(f" gc objects: {gc.get_objects()}")
            #for stat in snapshot_before_train[:10]:
                #print(stat)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            eval_bleu=None
            for line in process.stdout:
                sys.stdout.write(f"\r{line}".rstrip('\n'))
                sys.stdout.flush()
                #print(line, end='\r')
                #print(line, end="")
                if "EVAL_BLEU_RESULT:" in line:
                    match = re.search(r"EVAL_BLEU_RESULT:\s*(-?\d+\.\d+)", line)
                    if match:
                        eval_bleu = float(match.group(1))
                    else:
                        print("eval bleu not read properly")
            process.wait()

            log_dict = {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_train_epochs": num_train_epochs,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "encoder_layers": encoder_layers,
                "d_model": d_model,
                "eval_bleu": eval_bleu,
                "Out of Memory": False
                }
            wandb.log(log_dict)
            #then basically I want to do what pipeline does, train, eval, then get the bleu out of the eval and return the negative of the bleu,
            #because the optuna minimizes by default
            return (-1 * eval_bleu)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log_dict["Out of Memory"] = True
                wandb.log(log_dict)
                return float("inf")
            else:
                raise
    wandb_callback = WeightsAndBiasesCallback(metric_name="eval_bleu")
    study = optuna.create_study(direction="minimize")
    study.optimize(hyperparameter_search, n_trials=n_trials, callbacks=[wandb_callback])
    wandb.log(study.best_params)
    return


def main():

    ######################
    #CHANGE THIS BOOLEAN TO PERFORM A BASELINE OR AN EXPERIMENTAL SWEEP
    is_baseline = True
    ######################

    full_sweep_langs = ["so", "ba", "fo", "tt", "ga"]
    for src_lang in full_sweep_langs:
        #BASELINE
        if is_baseline:
            initial_checkpoint = f"facebook/nllb-200-distilled-600m"
            wandb.init(project=f"{src_lang}-en-baseline-tune-lr-bs-ne-wd_0_1")
            wandb_sweep(
                is_baseline=True,
                initial_checkpoint=initial_checkpoint,
                src_lang=src_lang,
                trg_lang='en',
                expdir="WandB_Baseline_Sweep_Test_0_1"
            )
            wandb.finish()
        #EXPERIMENTAL
        else:
            initial_checkpoint = f"./Experiment_1/{src_lang}-en_mode-baseline_train-True_eval-True/training_results"
            wandb.init(project=f"{src_lang}-en-experimental-tune-lr-bs-ne-wd-do-nl-dm_0_1")
            wandb_sweep(
                is_baseline=False,
                initial_checkpoint=initial_checkpoint,
                src_lang=src_lang,
                trg_lang='en',
                expdir="WandB_Experimental_Sweep_Test_0_1"
            )
            wandb.finish()


if __name__ == "__main__":
    main()