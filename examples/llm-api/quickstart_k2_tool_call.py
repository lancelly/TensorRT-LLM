from openai import OpenAI
from transformers import AutoTokenizer

def generate_prompt(model_name, messages, tools):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools,
        add_generation_prompt=True,
    )
    return text

# ref: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
def extract_tool_call_info(tool_call_rsp: str):
    if '<|tool_calls_section_begin|>' not in tool_call_rsp:
        # No tool calls
        return []
    import re
    pattern = r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
    
    tool_calls_sections = re.findall(pattern, tool_call_rsp, re.DOTALL)
    
    # Extract multiple tool calls
    func_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>"
    tool_calls = []
    for match in re.findall(func_call_pattern, tool_calls_sections[0], re.DOTALL):
        function_id, function_args = match
        # function_id: functions.get_weather:0
        function_name = function_id.split('.')[1].split(':')[0]
        tool_calls.append(
            {
                "id": function_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": function_args
                }
            }
        )  
    return tool_calls
 
def get_tools():
    # Collect the tool descriptions in tools
    return [{
        "type": "function",
        "function": {        
            "name": "get_weather", 
            "description": "Get weather information. Call this tool when the user needs to get weather information", 
                "parameters": {
                    "type": "object",
                    "required": ["city"], 
                    "properties": { 
                        "city": { 
                            "type": "string", 
                            "description": "City name", 
                    }
                }
            }
        }
        }]

if __name__ == "__main__":
    model_name = "moonshotai/Kimi-K2-Instruct"
    messages = [
    {"role": "user", "content": "What's the weather like in Beijing today? Let's check using the tool."}
    ]
    tools = get_tools()

    prompt = generate_prompt(model_name, messages, tools)
    print(f"prompt: {prompt}\n")
    
    # start trt-llm server before running this script
    client = OpenAI(
        api_key = "tensorrt_llm",
        base_url = "http://localhost:8000/v1",
    )
    
    response = client.completions.create(
        model="Kimi-K2-Instruct",
        prompt=prompt,
        max_tokens=2048,
    )
    print(f"response: {response}\n")

    tool_calls = extract_tool_call_info(response.choices[0].text)
    print(f"tool_calls: {tool_calls}\n")
 

