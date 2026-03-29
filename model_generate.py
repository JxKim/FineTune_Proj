from gc import enable

from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",type=str,default="model/Qwen3-0.6B-Base",help="模型路径")
parser.add_argument("--tokenizer_name",type=str,default=None,help="模型路径")
parser.add_argument("--prompt",type=str,default="你好",help="提示词")
parser.add_argument("--use_quantization",type=bool,default=False,help="是否使用量化")
args = parser.parse_args()
model_name = args.model_name
tokenizer_name = args.tokenizer_name
prompt = args.prompt
print('输入的prompt为：',prompt[:30])
print("模型路径为：",model_name)
use_quantization = args.use_quantization
tokenizer_name = tokenizer_name or model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,device_map="auto")
inputs = tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,return_tensors="pt",max_length=2048,enable_thinking=True)
print("得到的inputs为：",inputs)
input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs["attention_mask"].to("cuda")
if use_quantization:
    print("使用量化")
    from transformers import BitsAndBytesConfig
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=config)
    import torch
    print("量化模型后显存占用Allocated为：",torch.cuda.memory_allocated() / 1024 / 1024 / 1024 ,"GB")
    print("量化模型后显存占用Reserved为：",torch.cuda.memory_reserved() / 1024 / 1024 / 1024 ,"GB")
else:
    model = AutoModelForCausalLM.from_pretrained(model_name,dtype = torch.float16,device_map="auto")
    import torch
    print("显存占用Allocated为：",torch.cuda.memory_allocated() / 1024 / 1024 / 1024 ,"GB")
    print("显存占用Reserved为：",torch.cuda.memory_reserved() / 1024 / 1024 / 1024 ,"GB")
model.eval()


output_ids = model.generate(inputs=input_ids,attention_mask=attention_mask,max_new_tokens=5000,)[0][len(input_ids[0]):].tolist()
print('当前的output_ids位：',output_ids)
index = len(output_ids) - output_ids[::-1].index(151668)
thinking_content = tokenizer.decode(output_ids[:index],skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:],skip_special_tokens=True).strip("\n")
print("thinking content为：",thinking_content)
print("content为：",content)
