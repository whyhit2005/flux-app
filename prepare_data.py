from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import json
import os,sys,shutil
from tqdm import tqdm
from PIL import Image
import gc
import argparse
import warnings
from pathlib import Path
import re
import wd14_tagger
import logging
logging.basicConfig(level=logging.INFO)
from omegaconf import OmegaConf
import yaml

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="prepare data for training")
    parser.add_argument("cfg_file", type=str,
                        help="config file")
    return parser.parse_args(input_args)

def load_config_yaml(args):
    with open(args.cfg_file, "r") as f:
        cfg_data = yaml.safe_load(f)
        t_data = {}
        t_data.update(cfg_data["base"])
        t_data.update(cfg_data["prepare"])
        cfg_args = OmegaConf.create(t_data)
        return cfg_args
    
def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # 重複を排除
    image_paths.sort()
    return image_paths

def read_data(instance_dir):
    # create a list of (Pil.Image, path) pairs
    if not instance_dir.exists():
        print(f"Directory {instance_dir} does not exist")
        raise FileNotFoundError

    imgs_and_paths = []
    filelist = glob_images_pathlib(instance_dir, recursive=False)
    for file in filelist:
        imgs_and_paths.append((Image.open(file), file))
    return imgs_and_paths

# load the processor and the captioning model
def blip_prepare_data(imgs_and_paths, output_dir, caption_prompt):
    # captioning utility
    def caption_images(input_image):
        inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
        pixel_values = inputs.pixel_values

        generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=60)
        generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_caption
    
    with open(output_dir / 'metadata.jsonl', 'w') as outfile:
        for image, path in tqdm(imgs_and_paths, desc=f"{caption_prompt}"):
            caption = caption_prompt
            caption += ", "+caption_images(image).split("\n")[0]
            tarpath = output_dir / path.name
            shutil.copy(path, tarpath)
            entry = {"file_name":path.name, "prompt": caption}
            json.dump(entry, outfile)
            outfile.write('\n')


def wd14_prepare_data(imgs_and_paths, output_dir, caption_prompt, extra_black_words=None):
    wd14args.desc = caption_prompt.split(",")[0]
    tag_results = wd14_tagger.tag_images(wd14_model, wd14_all_tags, imgs_and_paths, wd14args)
    tag_black_list = [
    "eye", "eyes", "lip", "nose", "ear", "mouth", "teeth", "tongue", "neck",
    "blurry", "hair", "bald", "face", "skin", "head", "body",
    "buzz_cut", "mohawk", "ponytail",
    ]
    if extra_black_words is not None:
        tag_black_list.extend(extra_black_words)
    pstr = r"|".join(tag_black_list)
    pattern = re.compile(pstr, re.IGNORECASE)
    
    freqall = {}
    for img_path, tags in tag_results:
        tokens = tags.split(", ")
        for token in tokens:
            if token not in freqall:
                freqall[token] = 0
            freqall[token] += 1
    freqfile = output_dir.parent / "freqall.log"
    with open(freqfile, "w") as f:
        freqlist = sorted(freqall.items(), key=lambda x: x[1], reverse=True)
        for token, freq in freqlist:
            f.write(f"{token}: {freq}\n")
    
    freqs = {}
    tlist = []
    for img_path, tags in tag_results:
        tokens = tags.split(", ")
        out_tokens = [caption_prompt]
        for token in tokens:
            pres = pattern.search(token)
            if pres is not None:
                continue
            out_tokens.append(token)
            if token not in freqs:
                freqs[token] = 0
            freqs[token] += 1
        out_tag = ", ".join(out_tokens)
        tlist.append((img_path, out_tag))
        
    freqfile = output_dir.parent / "freqs.log"
    with open(freqfile, "w") as f:
        freqlist = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        for token, freq in freqlist:
            f.write(f"{token}: {freq}\n")

    with open(output_dir / 'metadata.jsonl', 'w') as outfile:
        for img_path, out_tag in tlist:
            shutil.copy(img_path, output_dir / img_path.name)
            entry = {"file_name":img_path.name, "prompt": out_tag}
            json.dump(entry, outfile)
            outfile.write('\n')

  
def main(cfg_args):
    if cfg_args.instance_image_dir is None:
        raise ValueError("instance_dir is not set")
    
    instance_dir = Path(cfg_args.instance_image_dir)
    work_dir = Path(cfg_args.work_dir)

    dir_list = []
    if cfg_args.subfolders:
        dir_list = list(instance_dir.iterdir())
    else:
        dir_list = [instance_dir]
    dir_list = sorted(dir_list, key=lambda x: x.name)
    
    instance = cfg_args.token_abstraction
    class_prompt = cfg_args.class_prompt
    caption_prefix = cfg_args.caption_prefix
    caption_prompt = f"{caption_prefix} {instance} {class_prompt}"
    if cfg_args.additional_caption is not None:
        caption_prompt += f", {cfg_args.additional_caption}"
    for tdir in dir_list:
        logging.info(f"Processing {tdir.name}")
        imgs_and_paths = read_data(tdir)
        image_dir = work_dir / tdir.name / cfg_args.images_dir_name
        if cfg_args.init_new and os.path.exists(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        if cfg_args.interrogator == "blip":
            blip_prepare_data(imgs_and_paths, image_dir, caption_prompt)
        elif cfg_args.interrogator == "wd14":
            wd14_prepare_data(imgs_and_paths, image_dir, caption_prompt, extra_black_words=cfg_args.extra_black_words)
    return 0

if __name__ == "__main__":
    args = parse_args()
    cfg_args = load_config_yaml(args)
    
    if cfg_args.interrogator == "blip":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco", torch_dtype=torch.float16)
        blip_model = blip_model.to(device)
        ret = main(cfg_args)
    elif cfg_args.interrogator == "wd14":
        wd14args = wd14_tagger.ImageTaggerArgs()
        wd14args.undesired_tags = "1girl,1boy,1women,1man,1person,child,solo"
        wd14_model, wd14_all_tags = wd14_tagger.load_model_and_tags(wd14args)
        ret = main(cfg_args)
    else:
        raise NotImplementedError("Integrator not implemented")
    sys.exit(ret)