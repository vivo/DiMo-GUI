# TODO: If you are running this code for the first time, follow the instructions from this link to fix the codebase first: https://github.com/OpenGVLab/InternVL/issues/405#issuecomment-2357514453
import json
import os
import re
import tempfile
import base64
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from models.utils.region_traverser import RegionTraverser
from utils.utils import *


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, Image.Image):
        image = image_file.convert('RGB')
    elif isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        raise ValueError("Input image must be a PIL Image or a file path string.")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str


def pred_2_point(s):
    # Extract the pattern [[number,number]]
    matches = re.findall(r'\[\[(\d+),(\d+)\]\]', s)
    
    # Convert extracted matches to floats
    floats = [(float(x), float(y)) for x, y in matches]
    
    # Return only the last match
    if floats:
        return list(floats[-1])  # Return the last pair as a flat list
    else:
        return None  # Return None if no matches are found
    
def extract_label(answer_str):
    return '1'
    """
    从字符串中提取最后的数字标签，例如 "answer: [[497,315]]: 1" -> 1
    """
    match = re.search(r":\s*(-?\d+)\s*$", answer_str)
    if match:
        
        return str(match.group(1))
    else:
        # raise ValueError("No label found at the end of the string.")
        return '1'

def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


class OSAtlas4BModel():
    def load_model(self, model_name_or_path="OS-Copilot/OS-Atlas-Base-4B", device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            device_map=device, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Setting default generation config
        # self.generation_config = GenerationConfig.from_pretrained("OS-Copilot/OS-Atlas-Base-4B", trust_remote_code=True).to_dict()
        self.generation_config = dict()
        self.set_generation_config(
            max_length=4096,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        # self.model.generation_config = GenerationConfig(**self.generation_config)

    def calculate_relative_distance(self, normalized_point, image_width, image_height):
        import math
        """
        计算归一化坐标与图像中心的相对距离。

        参数:
        - normalized_point: tuple，归一化坐标 (x, y)，值范围在 [0, 1]。
        - image_width: int，图像的宽度。
        - image_height: int，图像的高度。

        返回:
        - relative_distance: float，相对距离，归一化到 [0, 1]。
        """
        # 图像中心的归一化坐标
        center_x = 0.5
        center_y = 0.5

        # 归一化坐标
        normalized_x, normalized_y = normalized_point

        # 计算欧几里得距离
        distance = math.sqrt((normalized_x - center_x) ** 2 + (normalized_y - center_y) ** 2)

        # 最大可能距离（图像对角线的一半，归一化后为 sqrt(0.5^2 + 0.5^2)）
        max_distance = math.sqrt(0.5 ** 2 + 0.5 ** 2)

        # 计算相对距离
        relative_distance = distance / max_distance

        return relative_distance


    def get_coordinate_prediction(self, image_path, instruction, prompt):
        pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).cuda()
        prompt_origin = 'In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions(with point).\n"{}"'
        full_prompt = prompt_origin.format(instruction)
        response, history = self.model.chat(self.tokenizer, pixel_values, full_prompt, self.generation_config, history=None, return_history=True)

        # result_dict = {
        #     "result": "positive",
        #     "format": "x1y1x2y2",
        #     "raw_response": response,
        #     "bbox": None,
        #     "point": None,
        #     "iterations": 1
        # }

        click_point = pred_2_point(response)
        # click_point = [x / 1000 for x in click_point] if click_point else None
        print(click_point)
        # result_dict["point"] = click_point  # can be none
        
        return response, None, click_point

    def iterative_narrowing(self, instruction, image_path, subset,  prompt, modality, max_iter, threshold, verbose=True, dim = 999):
        target = instruction
        image = Image.open(image_path)
        width, height = image.size
        current_region = image.copy()
        cropped_image = image.copy()

        # 创建输出目录
        output_dir = os.path.join('./output/osatlas4b', subset, target)
        os.makedirs(output_dir, exist_ok=True)

        traverser = RegionTraverser(current_region)
        if verbose: 
            print(target)

        # 参数设置
        k = max_iter # 最大迭代轮数
        threshold = threshold  # 截断阈值
        # render_crosshair_center(current_region).show()

        response = None
        bbox = None
        point = None
        prev_dis = 1.0
        iteration = 0

        log_file_path = os.path.join(output_dir, "confidence_log.txt")  # 定义日志文件路径

        with open(log_file_path, "w") as log_file:  # 打开文件以写入模式
            # log_file.write(f"prompt: {prompt}\n")  # 写入提示信息
            for i in range(k):

                # print("caption:", caption)
                print("iteration: ", i)

                # caption = self.generate_image_caption(current_region)
                # # 在 prompt 和 caption 之间添加提示语
                # full_prompt = f"{prompt} Caption of the image: {caption}"
                # print("full_prompt:", full_prompt)

                image_prompt = current_region.resize((dim, dim))


                # 使用完整的 prompt 输入模型
                response, bbox, prediction_coord = self.get_coordinate_prediction(image_prompt, target, prompt)

                # attention_path = os.path.join(output_dir, f"attention_map_step_{i}.png")
                # self.save_attention_map(attentions, attention_path)

                if prediction_coord is None:
                    return response, bbox, prediction_coord, iteration

                pred_x, pred_y = prediction_coord


                if verbose: 
                    print("middle coord:", prediction_coord)
                if i != k-1:
                    traverser.consume_coordinate(pred_x, pred_y)
                    result_image = traverser.get_highlighted_image()
                    cropped_image = traverser.get_cropped_image()
                    current_region = cropped_image.resize((dim, dim))
                    # print("current_region:", current_region.size)
                    # print("current_region:", current_region)

                    # 保存中间结果图像
                    result_image_path = os.path.join(output_dir, f"step_{i}_highlighted.png")
                    result_image.convert("RGB").save(result_image_path)
                    print("Saved highlighted image to:", result_image_path)

                # 保存当前区域图像
                # current_region_path = os.path.join(output_dir, f"step_{i}_current_region.png")
                # current_region.convert("RGB").save(current_region_path)

                
                # 保存当前区域原尺寸图像
                cropped_image_path = os.path.join(output_dir, f"step_{i}_cropped_image.png")
                cropped_image.convert("RGB").save(cropped_image_path)

                iteration = i + 1


                prediction_coord = (pred_x / dim, pred_y / dim)  # 归一化相对坐标

                dis = self.calculate_relative_distance(prediction_coord, cropped_image.size[0], cropped_image.size[1])
                print("relative distance:", dis)
                

                if dis < threshold and prev_dis < threshold:
                    break
                
                prev_dis = dis

        final_bbox = traverser.get_bounding_box()
        print("final_bbox:", final_bbox)

        last_porp_x = pred_x / dim
        last_porp_y = pred_y / dim
            
        
        delta_x = (final_bbox[2] - final_bbox[0]) * last_porp_x
        delta_y = (final_bbox[3] - final_bbox[1]) * last_porp_y

        # 绝对坐标
        x, y = final_bbox[0] + delta_x, final_bbox[1] + delta_y
        if verbose:
            # 保存最终的十字准线图像
            crosshair_current_path = os.path.join(output_dir, "final_crosshair_current_region.png")
            render_crosshair(current_region, pred_x, pred_y).convert("RGB").save(crosshair_current_path)

            crosshair_full_image_path = os.path.join(output_dir, "final_crosshair_full_image.png")
            render_crosshair(image, x, y).convert("RGB").save(crosshair_full_image_path)

        # 返回归一化的坐标
        # print("image size:", image.size)
        # point = (x / width, y / height)
        point = (round(x / width, 4), round(y / height, 4))

        return response, bbox, point, iteration


    def select_answer(self, image_path, prompt):
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image",
        #                 "image": image_path,
        #             },
        #             {"type": "text", "text": decision_prompt},
        #         ],
        #     }
        # ]
        # # Preparation for inference
        # text_input = self.processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        # image_inputs, video_inputs = process_vision_info(messages)
        # inputs = self.processor(
        #     text=[text_input],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # inputs = inputs.to(self.device)
        # generated_ids = self.model.generate(**inputs)

        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # answer = self.processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        # )[0]

        # answer = answer.replace("<|im_end|>", "").strip()
        # 加载图像，准备 pixel_values
        pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).cuda()

        # 模型推理
        response, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            self.generation_config,
            history=None,
            return_history=True
        )

        print("Model response:", response)
        answer = extract_label(response)
        print("Selected answer:", answer)

        return answer



    def ground_only_positive_iterative_conquer(self, instruction, subset, image, max_iter, threshold):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        print(image_path)
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."


        # todo: divide and conquer
        prompt_origin = 'In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions(with point).\n"{}"'
        # text_prompt_origin = 'In this UI screenshot, please find the TEXT element corresponding to the command "{}" (with bbox). Only focus on text content, ignore icons.'
        # icon_prompt_origin = 'In this UI screenshot, please find the ICON element corresponding to the command "{}" (with bbox). Only focus on graphical icons, ignore pure text.'
        text_prompt_origin = 'In the screenshot of this web page, please give me the coordinates of the TEXT element I want to click on according to my instructions(with point), ignore icons.\n"{}"'
        icon_prompt_origin = 'In the screenshot of this web page, please give me the coordinates of the ICON element I want to click on according to my instructions(with point), ignore text.\n"{}"'

        text_prompt = text_prompt_origin.format(instruction)
        icon_prompt = icon_prompt_origin.format(instruction)
        
        response_text, bbox_text, point_text, text_iteration = self.iterative_narrowing(instruction=instruction, image_path=image_path, subset=subset + "-text", prompt=text_prompt, modality="text", max_iter=max_iter, threshold=threshold)
        response_icon, bbox_icon, point_icon, icon_iteration = self.iterative_narrowing(instruction=instruction, image_path=image_path, subset=subset + "-icon", prompt=icon_prompt, modality="icon", max_iter=max_iter, threshold=threshold)


        # todo: select the best answer
        decision_prompt_template = (
            "You are given a UI screenshot and a user command: \"{instruction}\".\n"
            "There are two candidate UI elements:\n\n"
            "Candidate 1 (text-based): {point_text}\n"
            "Candidate 2 (icon-based): {point_icon}\n\n"
            "Which candidate better matches the command?\n"
            "Answer only with the number 1 or 2.\n"
            "Do not include any other text, explanation, or symbols. Just reply with '1' or '2'."
        )


        decision_prompt = decision_prompt_template.format(
            instruction=instruction,
            point_text=point_text,
            point_icon=point_icon,
        )

        print(decision_prompt)
        answer = self.select_answer(image_path, decision_prompt)
        print("answer:", answer)
        if answer == '1':
            response = response_text
            bbox = bbox_text
            point = point_text
        elif answer == '2':
            response = response_icon
            bbox = bbox_icon
            point = point_icon
        else:
            raise ValueError("Invalid answer. Answer must be either '1' or '2'.")


        ## vanilla osatlas
        # response, bbox, point = self.get_model_response(instruction, image_path)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": bbox,
            "point": point,
            "iterations": text_iteration + icon_iteration,
        }
        print("result_dict:", result_dict)
        
        return result_dict

    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).cuda()

        prompt_origin = 'In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions(with point).\n"{}"'
        full_prompt = prompt_origin.format(instruction)
        response, history = self.model.chat(self.tokenizer, pixel_values, full_prompt, self.generation_config, history=None, return_history=True)
        # print(response)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None,
            "iterations": 1
        }


        click_point = pred_2_point(response)
        click_point = [x / 1000 for x in click_point] if click_point else None
        print(click_point)
        result_dict["point"] = click_point  # can be none
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).cuda()

        prompt_origin = 'In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions(with point).\n"{}"'
        full_prompt = prompt_origin.format(instruction)
        response, history = self.model.chat(self.tokenizer, pixel_values, full_prompt, self.generation_config, history=None, return_history=True)

        result_dict = {
            "result": None,
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        click_point = pred_2_point(response)
        click_point = [x / 1000 for x in click_point]
        result_dict["point"] = click_point  # can be none

        # set result status
        if result_dict["bbox"] or result_dict["point"]:
            result_status = "positive"
        elif "Target does not exist".lower() in response.lower():
            result_status = "negative"
        else:
            result_status = "wrong_format"
        result_dict["result"] = result_status

        return result_dict

