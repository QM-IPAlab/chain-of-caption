import argparse
import json
import logging
import math
import os
import signal
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from collections import defaultdict

eval_logger = logging.getLogger("lmms-eval")
# import sys;sys.path.append("llava-video")
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    raise ImportError(f"VILA is not installed. Please install VILA to use this model. Error: {e}")


import cv2
import re
import copy
from PIL import ImageDraw, ImageFont

def draw_text(pil_img, text):
    """
    Draw text on the image in black with a white background
    :param pil_img: PIL Image object.
    :param text: Text to draw on the image.
    """
    draw = ImageDraw.Draw(pil_img)
    text_width, text_height = draw.textbbox((0, 0), text)[2:4]
    width, height = pil_img.size
    x = (width - text_width) // 2
    y = height - text_height - 10  # 10 pixels from the bottom
    draw.rectangle([x, y, x + text_width, y + text_height], fill="white")
    draw.text((x, y), text, fill="black")

def draw_bounding_box_debug(pil_img, pred_bbox, gt_bbox):
    """
    Draws a bounding box on the image and saves the result.

    :param pil_img: PIL Image object.
    :param pred_bbox: Predicted bounding box coordinates in the format (x_min, y_min, x_max, y_max).
    :param gt_bbox: Ground truth bounding box coordinates in the format (x_min, y_min, x_max, y_max).
    """
    # Load the image
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Draw the bounding box
    # Convert bbox from relative to absolute coordinates
    height, width = image.shape[:2]
    x_min = int(pred_bbox[0] * width)
    y_min = int(pred_bbox[1] * height)
    x_max = int(pred_bbox[2] * width)
    y_max = int(pred_bbox[3] * height)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw the ground truth bounding box
    gt_x_min = int(gt_bbox[0] * width)
    gt_y_min = int(gt_bbox[1] * height)
    gt_x_max = int(gt_bbox[2] * width)
    gt_y_max = int(gt_bbox[3] * height)
    cv2.rectangle(image, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (255, 0, 0), 2)

def draw_bounding_box_list(pil_image, bbox_list, iterate_color=False):
    """
    Draws a bounding box on the image and saves the result.

    :param image_path: Path to the input image.
    :param bbox: Bounding box coordinates in the format (x_min, y_min, x_max, y_max).
    """
    # Load the image
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    if len(bbox_list) == 0:
        raise ValueError("No bounding boxes provided to draw.")
    elif len(bbox_list) > 6:
        raise ValueError("Too many bounding boxes provided to draw. Maximum is 6.")

    # Draw the bounding box
    # Convert bbox from relative to absolute coordinates
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i, bbox in enumerate(bbox_list):
        height, width = image.shape[:2]
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)
        if iterate_color:
            color = colors[i % len(colors)]
        else:
            color = (0, 255, 0)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def bounding_box_dim(pil_img, bbox):
    """
    Dim the bounding box area in the image.
    """
    if not pil_img or not bbox:
        return None

    # Get the bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Dim the bounding box area
    img_array = np.array(pil_img)
    img_array[y_min:y_max, x_min:x_max] = img_array[y_min:y_max, x_min:x_max] * 0.5

    return Image.fromarray(img_array)

def extract_bbox_list(response):
    response = response.replace(",]", "]")  # Fix any trailing commas in lists
    response_lines = response.split("\n")
    bbox_pattern = re.compile(r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]")
    bbox_list = []
    for line in response_lines:
        match = bbox_pattern.search(line)
        if match:
            bbox = tuple(float(coord) for coord in match.groups())
            if len(bbox) == 4:
                bbox_list.append(bbox)
            else:
                print("Invalid bounding box format in response:", line)
    return bbox_list
    
def extract_bbox_coordinates(text):
    """
    Extracts bounding box coordinates from a text string.

    :param text: Text containing bounding box coordinates in the format (x_min, y_min, x_max, y_max).
    :return: A list of bounding box coordinates as floats.
    """
    # Remove any unwanted characters and split the string
    text = text.replace(",]", "]")
    text = text.replace("(", "[")
    text = text.replace(")", "]")
    bbox_pattern = re.compile(r"\[([0-9.]+), ([0-9.]+), ([0-9.]+), ([0-9.]+)\]")
    match = bbox_pattern.search(text)
    if match:
        return [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]
    else:
        return [0.0, 0.0, 0.0, 0.0]  # Return a default value if no match is found

def infer(img_orig, model, contexts, desc_mode='all', draw_bbox=False, coc=False, coc_trials=5, dim_false=False, desc_loc=False):

    # Get global image description
    desc_prompts = [img_orig]
    if desc_mode is not 'none':
        if desc_mode == 'all':
            desc_prompts.append("Detect 5 object and output their description in a numbered list. " +
                            "Keep each description under 10 words. ")
        elif type(desc_mode) == int:
            desc_prompts.append("Detect {} objects and output their description and bounding box in floating point numbers. ".format(desc_mode)+
                            "Format the output as a numbered list where each line contains exactly one description and one bounding box "+
                            "Bounding box coordinates are specified in the format [top-left x, top-left y, bottom-right x, bottom-right y]. "+
                            "with values between 0.0 and 1.0. "+
                            "Keep each line under 100 characters.  Detected objects: ")
        desc = model.generate_content(desc_prompts, response_format=None)
        print("Description: "+desc)
    # Draw bounding box detected from global description
    if draw_bbox:
        bbox_list = extract_bbox_list(desc)
        if len(bbox_list) > 0:
            img = draw_bounding_box_list(img_orig, bbox_list, iterate_color=True)
        else:
            eval_logger.warning(f"No bounding boxes found in the description: {desc}. Using original image without bounding boxes.")
    else:
        img = img_orig

    # Detect target
    if not coc:
        target_prompts = [img]
        if desc_mode is not "none" and draw_bbox is False:
            target_prompts.append("Objects in the image: " + desc) # add general image description
        target_prompts.append(contexts)
        outputs = model.generate_content(target_prompts, response_format=None)
    else:
        trial = 1
        orig_prompt_text = contexts.split(":")[-1].strip()
        if desc_mode == 'none':
            target_prompts = [img, contexts]
            insert_idx = 1
        else:
            target_prompts = [img, "Objects in the image: " + desc , contexts]
            insert_idx = 2
        crop_desc_old = ''
        while trial < coc_trials:
            # initial pred
            outputs = model.generate_content(target_prompts, response_format=None)
            pred_bbox = extract_bbox_coordinates(outputs)

            # crop image with pred_bbox
            img_w, img_h = img.size
            img_crop = img.crop((pred_bbox[0]*img_w, pred_bbox[1]*img_h, pred_bbox[2]*img_w, pred_bbox[3]*img_h))

            # describe image crop
            crop_prompts = [img_crop, 'Describe the image in under 10 words.']
            crop_desc = model.generate_content(crop_prompts, response_format=None)

            if crop_desc == crop_desc_old:
                break
            else:
                crop_desc_old = crop_desc

            # ask llm if these two are the same
            qa = verify_crop(img_crop, orig_prompt_text, model)
            if 'yes' in qa.lower():
                return str(pred_bbox)
            else:
                if type(desc_mode) == int:
                    offset = desc_mode
                else:
                    offset = 0
                target_prompts[1] = target_prompts[1] + "\n" + "{}. {} {}".format(trial+offset, crop_desc, str(pred_bbox))
                if dim_false:
                    img = bounding_box_dim(img, pred_bbox)
                    target_prompts[0] = img
                trial += 1
                insert_idx += 1
            
    return outputs

def infer_crop(img_orig, model, contexts):
    # Detect target
    target_prompts = [img_orig, contexts]
    img_w, img_h = float(img_orig.size[0]), float(img_orig.size[1])
    outputs = model.generate_content(target_prompts, response_format=None)

    # crop image with pred_bbox
    orig_pred_bbox = extract_bbox_coordinates(outputs)
    margin = 0.5
    crop_min_x = max(0.0, min(1.0, orig_pred_bbox[0]-(orig_pred_bbox[2]-orig_pred_bbox[0])*margin))*img_w
    crop_min_y = max(0.0, min(1.0, orig_pred_bbox[1]-(orig_pred_bbox[3]-orig_pred_bbox[1])*margin))*img_h
    crop_max_x = max(0.0, min(1.0, orig_pred_bbox[2]+(orig_pred_bbox[2]-orig_pred_bbox[0])*margin))*img_w
    crop_max_y = max(0.0, min(1.0, orig_pred_bbox[3]+(orig_pred_bbox[3]-orig_pred_bbox[1])*margin))*img_h
    img_crop = img_orig.crop((crop_min_x, crop_min_y, crop_max_x, crop_max_y))
    crop_w, crop_h = float(img_crop.size[0]), float(img_crop.size[1])

    # Detect target again
    target_prompts = [img_crop, contexts]
    outputs = model.generate_content(target_prompts, response_format=None)

    # Retrieve the bounding box in the original image
    crop_pred_bbox = extract_bbox_coordinates(outputs)
    new_pred_bbox = [max(0.0, min(1.0, crop_pred_bbox[0]*crop_w/img_w + crop_min_x/img_w)),
                    max(0.0, min(1.0, crop_pred_bbox[1]*crop_h/img_h + crop_min_y/img_h)),
                    max(0.0, min(1.0, crop_pred_bbox[2]*crop_w/img_w + crop_min_x/img_w)),
                    max(0.0, min(1.0, crop_pred_bbox[3]*crop_h/img_h + crop_min_y/img_h))]

    outputs = str(new_pred_bbox)
    return outputs

def compare_desc(input_desc, pred_desc, model):
    target_prompts = ['Object A: {} \n'.format(input_desc),
                        'Object B: {} \n'.format(pred_desc),
                    'Object A is a general description of a type of object. Is object B part of the type of object described in object A? '
                        'Explain your reasoning and give your answer in one word: Yes/No.']
    qa = model.generate_content(target_prompts, response_format=None)
    return qa

def verify_crop(crop_image, text, model):
    target_prompts = [crop_image, ' Does the image match the following description?: {}. '.format(text),
                      'Answer in one word: Yes/No.']
    qa = model.generate_content(target_prompts, response_format=None)
    return qa


@register_model("vila")
class VILA(lmms):
    """
    VILA Model
    """

    def __init__(
        self,
        pretrained: str = "Efficient-Large-Model/VILA1.5-40b",
        max_frames_num: Optional[int] = 100,
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="hermes-2",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        video_decode_backend="decord",
        desc_mode: Union[str, int] = "none",  # "none", "all", or int (e.g. 5 for top-k object descriptions)
        draw_bbox: bool = False,
        coc: bool = False,
        crop_and_zoom: bool = False,
        debug_output_path: Optional[str] = None,  # if set, save debug visualizations; if None/empty, skip
        debug_draw_text: bool = False,
        debug_draw_bounding_box: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.max_frames_num = max_frames_num
        # self._config = AutoConfig.from_pretrained(self.pretrained)
        print("Attention: "+attn_implementation)
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, self.model_name, device_map=self.device_map, attn_implementation=attn_implementation)
        
        self.model.image_processor = self._image_processor

        self._config = self._model.config

        if self._tokenizer.pad_token_id is None:
            if "qwen" in self._tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self._tokenizer.pad_token_id = 151643

        self.video_decode_backend = video_decode_backend
        self.desc_mode = desc_mode
        self.draw_bbox = draw_bbox
        self.coc = coc
        self.crop_and_zoom = crop_and_zoom
        self.debug_output_path = debug_output_path or None
        self.debug_draw_text = debug_draw_text
        self.debug_draw_bounding_box = debug_draw_bounding_box
        self.model.eval()
        # self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def load_video(self, video_path, max_frames_num):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            frame_idx = np.linspace(0, total_frame_num - 2, max_frames_num, dtype=int)
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            return [Image.fromarray(img) for img in spare_frames]
        except Exception as e:
            eval_logger.error(f"Failed to load video {video_path} with error: {e}")
            return [Image.new("RGB", (448, 448), (0, 0, 0))] * max_frames_num

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            for visual in visuals:
                video = self.load_video(visual, self.max_frames_num)
                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                videos.append(video)

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # import pdb; pdb.set_trace()

            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=videos, modalities="video")

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Extract prompt text from the contexts
            try:
                prompt_text = contexts.split(":")[-1].strip()  # Assuming the last part after ':' is the prompt text
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)

                if self.crop_and_zoom:
                    outputs = infer_crop(visuals[0], self.model, contexts)
                else:
                    outputs = infer(
                        visuals[0],
                        self.model,
                        contexts,
                        desc_mode=self.desc_mode,
                        draw_bbox=self.draw_bbox,
                        coc=self.coc,
                    )  

            except Exception as e:
                eval_logger.error(f"Failed to generate content for {contexts} with error: {e}")
                outputs = "[0.0, 0.0, 0.0, 0.0]"  # Default output if generation fails
            
            # visualise bounding boxes
            pred_bbox = extract_bbox_coordinates(outputs)
            gt_bbox = self.task_dict[task][split][doc_id]["bbox"]
            print(f"Predicted BBox: {pred_bbox}, Ground Truth BBox: {gt_bbox}")

            if self.debug_output_path:
                if not os.path.exists(self.debug_output_path):
                    os.makedirs(self.debug_output_path)
                output_image_path = os.path.join(self.debug_output_path, f"{doc_id}_bbox.jpg")
                vis_img = visuals[0]
                if self.debug_draw_text:
                    draw_text(vis_img, prompt_text)
                if self.debug_draw_bounding_box:
                    draw_bounding_box_debug(vis_img, pred_bbox, gt_bbox)
                vis_img.save(output_image_path)

            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
