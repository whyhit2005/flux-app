import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

# from wd14 tagger
IMAGE_SIZE = 448

# SmilingWolf/wd-swinv2-tagger-v3 / wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
FILES_ONNX = ["model.onnx"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]


def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)
    return image


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
            # tensor = torch.tensor(image) # これ Tensor に変換する必要ないな……(;･∀･)
        except Exception as e:
            logger.error(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (image, img_path)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def load_model_and_tags(args):
    # model location is model_dir + repo_id
    # repo id may be like "user/repo" or "user/repo/branch", so we need to remove slash
    model_location = os.path.join(args.model_dir, args.repo_id.replace("/", "_"))

    # hf_hub_downloadをそのまま使うとsymlink関係で問題があるらしいので、キャッシュディレクトリとforce_filenameを指定してなんとかする
    # depreacatedの警告が出るけどなくなったらその時
    # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/issues/22
    if not os.path.exists(model_location) or args.force_download:
        os.makedirs(args.model_dir, exist_ok=True)
        logger.info(f"downloading wd14 tagger model from hf_hub. id: {args.repo_id}")
        files = FILES
        if args.onnx:
            files = ["selected_tags.csv"]
            files += FILES_ONNX
        else:
            for file in SUB_DIR_FILES:
                hf_hub_download(
                    args.repo_id,
                    file,
                    subfolder=SUB_DIR,
                    cache_dir=os.path.join(model_location, SUB_DIR),
                    force_download=True,
                    force_filename=file,
                )
        for file in files:
            hf_hub_download(args.repo_id, file, cache_dir=model_location, force_download=True, force_filename=file)
    else:
        logger.info("using existing wd14 tagger model")

    # モデルを読み込む
    if args.onnx:
        import torch
        import onnx
        import onnxruntime as ort

        onnx_path = f"{model_location}/model.onnx"
        logger.info("Running wd14 tagger with onnx")
        logger.info(f"loading onnx model: {onnx_path}")

        if not os.path.exists(onnx_path):
            raise Exception(
                f"onnx model not found: {onnx_path}, please redownload the model with --force_download"
                + " / onnxモデルが見つかりませんでした。--force_downloadで再ダウンロードしてください"
            )

        if "OpenVINOExecutionProvider" in ort.get_available_providers():
            # requires provider options for gpu support
            # fp16 causes nonsense outputs
            ort_sess = ort.InferenceSession(
                onnx_path,
                providers=(["OpenVINOExecutionProvider"]),
                provider_options=[{'device_type' : "GPU_FP32"}],
            )
        else:
            ort_sess = ort.InferenceSession(
                onnx_path,
                providers=(
                    ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else
                    ["ROCMExecutionProvider"] if "ROCMExecutionProvider" in ort.get_available_providers() else
                    ["CPUExecutionProvider"]
                ),
            )
    
    # label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")
    # 依存ライブラリを増やしたくないので自力で読むよ

    with open(os.path.join(model_location, CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = [row for row in reader]
        header = line[0]  # tag_id,name,category,count
        rows = line[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"

    rating_tags = [row[1] for row in rows[0:] if row[2] == "9"]
    general_tags = [row[1] for row in rows[0:] if row[2] == "0"]
    character_tags = [row[1] for row in rows[0:] if row[2] == "4"]

    if args.remove_underscore:
        rating_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in rating_tags]
        general_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in general_tags]
        character_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in character_tags]

    if args.tag_replacement is not None:
        # escape , and ; in tag_replacement: wd14 tag names may contain , and ;
        escaped_tag_replacements = args.tag_replacement.replace("\\,", "@@@@").replace("\\;", "####")
        tag_replacements = escaped_tag_replacements.split(";")
        for tag_replacement in tag_replacements:
            tags = tag_replacement.split(",")  # source, target
            assert len(tags) == 2, f"tag replacement must be in the format of `source,target` / タグの置換は `置換元,置換先` の形式で指定してください: {args.tag_replacement}"

            source, target = [tag.replace("@@@@", ",").replace("####", ";") for tag in tags]
            logger.info(f"replacing tag: {source} -> {target}")

            if source in general_tags:
                general_tags[general_tags.index(source)] = target
            elif source in character_tags:
                character_tags[character_tags.index(source)] = target
            elif source in rating_tags:
                rating_tags[rating_tags.index(source)] = target

    return ort_sess, [rating_tags, general_tags, character_tags]

def tag_images(sessions, all_tags, imgs_and_paths, args):
    ort_sess = sessions
    input = ort_sess.get_inputs()[0]
    rating_tags, general_tags, character_tags = all_tags
    # 画像を読み込む
    logger.info(f"found {len(imgs_and_paths)} images.")

    tag_freq = {}

    caption_separator = args.caption_separator
    stripped_caption_separator = caption_separator.strip()
    undesired_tags = args.undesired_tags.split(stripped_caption_separator)
    undesired_tags = set([tag.strip() for tag in undesired_tags if tag.strip() != ""])

    def run(image):
        if args.onnx:
            # if len(imgs) < args.batch_size:
            #     imgs = np.concatenate([imgs, np.zeros((args.batch_size - len(imgs), IMAGE_SIZE, IMAGE_SIZE, 3))], axis=0)
            probs = ort_sess.run(None, {input.name: image})[0]  # onnx output numpy
        else:
            probs = model(image, training=False)
            probs = probs.numpy()

        prob = probs[0]
        combined_tags = []
        rating_tag_text = ""
        character_tag_text = ""
        general_tag_text = ""

        # 最初の4つ以降はタグなのでconfidenceがthreshold以上のものを追加する
        # First 4 labels are ratings, the rest are tags: pick any where prediction confidence >= threshold
        for i, p in enumerate(prob[4:]):
            if i < len(general_tags) and p >= args.general_threshold:
                tag_name = general_tags[i]

                if tag_name not in undesired_tags:
                    tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                    general_tag_text += caption_separator + tag_name
                    combined_tags.append(tag_name)
            elif i >= len(general_tags) and p >= args.character_threshold:
                tag_name = character_tags[i - len(general_tags)]

                if tag_name not in undesired_tags:
                    tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                    character_tag_text += caption_separator + tag_name
                    if args.character_tags_first: # insert to the beginning
                        combined_tags.insert(0, tag_name)
                    else:
                        combined_tags.append(tag_name)

        # 最初の4つはratingなのでargmaxで選ぶ
        # First 4 labels are actually ratings: pick one with argmax
        if args.use_rating_tags or args.use_rating_tags_as_last_tag:
            ratings_probs = prob[:4]
            rating_index = ratings_probs.argmax()
            found_rating = rating_tags[rating_index]

            if found_rating not in undesired_tags:
                tag_freq[found_rating] = tag_freq.get(found_rating, 0) + 1
                rating_tag_text = found_rating
                if args.use_rating_tags:
                    combined_tags.insert(0, found_rating) # insert to the beginning
                else:
                    combined_tags.append(found_rating)

        # 先頭のカンマを取る
        if len(general_tag_text) > 0:
            general_tag_text = general_tag_text[len(caption_separator) :]
        if len(character_tag_text) > 0:
            character_tag_text = character_tag_text[len(caption_separator) :]

        tag_text = caption_separator.join(combined_tags)

        if args.debug:
            logger.info("")
            logger.info(f"{image_path}:")
            logger.info(f"\tRating tags: {rating_tag_text}")
            logger.info(f"\tCharacter tags: {character_tag_text}")
            logger.info(f"\tGeneral tags: {general_tag_text}")
            
        return tag_text

    tag_text_results = []
    for data in tqdm(imgs_and_paths, desc=args.desc):
        image, image_path = data
        image = preprocess_image(image)
        tag_text = run(image)
        tag_text_results.append((image_path, tag_text))

    if args.frequency_tags:
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        print("Tag frequencies:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    logger.info("done!")
    return tag_text_results

class ImageTaggerArgs:
    def __init__(self):
        self.repo_id = DEFAULT_WD14_TAGGER_REPO
        self.model_dir = "wd14_tagger_model"
        self.force_download = False
        self.thresh = 0.4
        self.general_threshold = self.thresh
        self.character_threshold = self.thresh
        self.recursive = False
        self.remove_underscore = False
        self.debug = False
        self.undesired_tags = ""
        self.frequency_tags = False
        self.onnx = True
        self.use_rating_tags = False
        self.use_rating_tags_as_last_tag = False
        self.character_tags_first = False
        self.caption_separator = ", "
        self.tag_replacement = None
        self.character_tag_expand = False
        self.desc = None
