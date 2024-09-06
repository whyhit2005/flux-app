import os, sys, shutil
from tqdm import tqdm
from pathlib import Path
import random
import json
import argparse
import warnings
import datetime
import logging
logging.basicConfig(level=logging.INFO)
from omegaconf import OmegaConf
import yaml

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("cfg_file", type=str,
                        help="config file")
    return parser.parse_args(input_args)

def load_config_yaml(args):
    cfg_file = args.cfg_file
    with open(cfg_file, "r") as f:
        cfg_data = yaml.safe_load(f)
        t_data = {}
        t_data.update(cfg_data["base"])
        t_data.update(cfg_data["train"])
        cfg = OmegaConf.create(t_data)
    return cfg

def train_process(cfg_args):
    if not cfg_args.work_dir:
        raise ValueError("work_dir is not set")
    if not cfg_args.task_name:
        raise ValueError("task_name is not set")
    
    work_dir = Path(cfg_args.work_dir)
    dirlist = []
    if cfg_args.subfolders:
        dirlist = list(work_dir.iterdir())
    else:
        dirlist = [work_dir]
    dirlist = sorted(dirlist, key=lambda x: x.name)

    for tdir in dirlist:
        token_abstraction = cfg_args.token_abstraction
        model_dir = tdir/ f"{cfg_args.model_dir_name}"/ f"{cfg_args.task_name}"
        images_dir = tdir / cfg_args.images_dir_name
        logging_dir = tdir / "logs"
        
        if cfg_args.init_new and os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        if cfg_args.init_new and os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)
        
        cmdfile = model_dir / "cmd.txt"
        donefile = model_dir / "done.txt"
        if os.path.exists(donefile) and model_dir.glob("*safetensors"):
            logging.info(f"{token_abstraction} already done, skip")
            continue
        
        instance_prompt = f"photo of a {cfg_args.token_abstraction} {cfg_args.class_prompt}"
        if cfg_args.validation_prompt and instance_prompt:
            validation_prompt = f'{instance_prompt}, {cfg_args.validation_prompt}'
            validation_epochs = cfg_args.num_train_epochs // 10
        else:
            validation_prompt = None
            validation_epochs = 0
        
        cmdstr = f'accelerate launch train_dreambooth_lora_flux.py \\\n'
        cmdstr += f' --pretrained_model_name_or_path={cfg_args.pretrained_model_name_or_path} \\\n'
        # cmdstr += f' --variant=fp16 \\\n'
        cmdstr += f' --dataset_name={images_dir} \\\n'
        cmdstr += f' --instance_prompt=\"{instance_prompt}\" \\\n'
        if validation_prompt:
            cmdstr += f' --validation_prompt=\"{validation_prompt}\" \\\n'
            cmdstr += f' --validation_epochs={validation_epochs} \\\n'
        cmdstr += f' --output_dir={model_dir} \\\n'
        cmdstr += f' --logging_dir={logging_dir} \\\n'
        cmdstr += f' --caption_column="prompt" \\\n'
        cmdstr += f' --mixed_precision="bf16" \\\n'
        cmdstr += f' --resolution=1024 \\\n'
        cmdstr += f' --train_batch_size={cfg_args.train_batch_size} \\\n'
        cmdstr += f' --repeats=1 \\\n'
        cmdstr += f' --report_to="wandb" \\\n'
        cmdstr += f' --gradient_accumulation_steps={cfg_args.gradient_accumulation_steps} \\\n'
        cmdstr += f' --gradient_checkpointing \\\n'
        if cfg_args.prodigy:
            cmdstr += f' --learning_rate=1.0 \\\n'
            cmdstr += f' --text_encoder_lr=1.0 \\\n'
            cmdstr += f' --optimizer="prodigy" \\\n'
            cmdstr += f' --prodigy_safeguard_warmup=True \\\n'
            cmdstr += f' --prodigy_use_bias_correction=True \\\n'
            cmdstr += f' --lr_scheduler="constant" \\\n'
            cmdstr += f' --lr_warmup_steps=0 \\\n'
        else:
            cmdstr += f' --learning_rate=1e-4 \\\n'
            cmdstr += f' --text_encoder_lr=1e-5 \\\n'
            cmdstr += f' --optimizer="Adamw" \\\n'
            cmdstr += f' --lr_scheduler="cosine" \\\n'
            cmdstr += f' --lr_warmup_steps=0 \\\n'
        cmdstr += f' --adam_weight_decay=0.01 \\\n'
        cmdstr += f' --adam_beta1=0.9 --adam_beta2=0.99 \\\n'
        cmdstr += f' --rank={cfg_args.rank} \\\n'
        cmdstr += f' --max_train_steps={cfg_args.max_train_steps} \\\n'
        cmdstr += f' --checkpoints_total_limit=1 \\\n'
        cmdstr += f' --checkpointing_steps={cfg_args.checkpointing_steps} \\\n'
        cmdstr += f' --resume_from_checkpoint=latest \\\n'
        cmdstr += f' --allow_tf32 \\\n'
        cmdstr += f' --seed={cfg_args.seed}'
        
        with open(cmdfile, "w") as f:
            f.write(cmdstr)
        start_time = datetime.datetime.now()
        ret = os.system(cmdstr)
        if ret != 0:
            warnings.warn(f"{token_abstraction} Lora training failed")
            return ret
        end_time = datetime.datetime.now()
        with open(donefile, "w") as f:
            f.write("done\n")
            f.write(f"start time: {start_time}\n")
            f.write(f"end time: {end_time}\n")
            f.write(f"duration: {end_time-start_time}\n")
            f.write(f"cmd: {cmdstr}\n")
    return 0

def infer_after_train(args):
    cmdstr = f'python sdxl_inference.py {args.cfg_file}'
    ret = os.system(cmdstr)
    return ret

if __name__ == "__main__":
    args = parse_args()
    cfg_args = load_config_yaml(args)
    ret = train_process(cfg_args)
    # if ret == 0:
    #     ret = infer_after_train(args)
    # else:
    #     warnings.warn("Training failed")
    sys.exit(ret)
