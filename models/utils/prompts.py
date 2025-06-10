PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT = '''Please generate the next move according to the UI screenshot and task instruction. You will be presented with a screenshot image. Also you will be given each bounding box's description in a list. To complete the task, You should choose a related bbox to click based on the bbox descriptions. 
Task instruction: {}. 
Here is the list of all detected bounding boxes by IDs and their descriptions: {}. Keep in mind the description for Text Boxes are likely more accurate than the description for Icon Boxes. 
Requirement: 1. You should first give a reasonable description of the current screenshot, and give a short analysis of how can the user task be achieved. 2. Then make an educated guess of bbox id to click in order to complete the task based on the bounding boxes descriptions. 3. Your answer should follow the following format: {{"Analysis": xxx, "Click BBox ID": "y"}}. Do not include any other info. Some examples: {}. The task is to {}. Retrieve the bbox id where its description matches the task instruction. Now start your answer:'''

# PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT_v1 = "The instruction is to {}. \nHere is the list of all detected bounding boxes by IDs and their descriptions: {}. \nKeep in mind the description for Text Boxes are likely more accurate than the description for Icon Boxes. \n Requirement: 1. You should first give a reasonable description of the current screenshot, and give a step by step analysis of how can the user task be achieved. 2. Then make an educated guess of bbox id to click in order to complete the task using both the visual information from the screenshot image and the bounding boxes descriptions. 3. Your answer should follow the following format: {{'Analysis': 'xxx', 'Click BBox ID': 'y'}}. Please do not include any other info."
PROMPT_TEMPLATE_SEECLICK_PARSED_CONTENT_v1 = "The instruction is to {}. \n" \
"Here is the list of all detected bounding boxes by IDs and their descriptions: {}. \n" \
"Keep in mind the description for Text Boxes are likely more accurate than the description for Icon Boxes. \n " \
"Requirement: " \
"1. You should first give a reasonable description of the current screenshot, and give a some analysis of how can the user instruction be achieved by a single click. " \
"2. Then make an educated guess of bbox id to click in order to complete the task using both the visual information from the screenshot image and the bounding boxes descriptions. REMEMBER: the task instruction must be achieved by one single click. " \
"3. Your answer should follow the following format: {{'Analysis': 'xxx', 'Click BBox ID': 'y'}}. Please do not include any other info."

# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
COMPUTER_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

MOBILE_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""