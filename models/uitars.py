import torch
from transformers import AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import json
import re
import os
from PIL import Image

import base64
from openai import OpenAI


import ast
def parse_string(input_string):
    try:
        box = eval(input_string)
        return box
    except:
        if 'Action:' in input_string:
            action_str = input_string.split('Action:')[1]
            try:
                result = parse_function_call(action_str)
                parameters = result['parameters']
                box = parameters.get('start_box',None)
            except:
                action_str,box = None,None
            return eval(box) if box is not None else None
        else:
            return None


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name



client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="empty",
)


class UiTarsModel():
    def load_model(self, model_name_or_path=""):
        pass

    def set_generation_config(self, **kwargs):
        pass

    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""
        
        prompt_origin = format(GROUNDING_DOUBAO, instruction=instruction)
        response_ = client.chat.completions.create(
            model="ui-tars",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
                        {"type": "text", "text": prompt_origin}
                    ],
                },
            ],
            frequency_penalty=1,
            max_tokens=128,
        )
        response = response_.choices[0].message.content

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        
        box = parse_string(response)
        if type(box)==tuple:
            box = [int(x) / 1000 for x in box] if box is not None else None
        result_dict["point"] = box
        return result_dict


    def ground_allow_negative(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        prompt_origin = r"""Output only the coordinate of one point in your response. What element matches the following task: """
        full_prompt = prompt_origin + instruction
        response_ = client.chat.completions.create(
            model="ui-tars",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
                        {"type": "text", "text": full_prompt}
                    ],
                },
            ],
            frequency_penalty=1,
            max_tokens=128,
        )
        response = response_.choices[0].message.content

        result_dict = {
            "result": None,
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }
        box = parse_string(response)
        if type(box)==tuple:
            box = [int(x) / 1000 for x in box] if box is not None else None
        result_dict["point"] = box
        if result_dict["bbox"] or result_dict["point"]:
            result_status = "positive"
        elif "Target does not exist".lower() in response.lower():
            result_status = "negative"
        else:
            result_status = "wrong_format"
        result_dict["result"] = result_status

        return result_dict

