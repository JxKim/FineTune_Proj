"""
使用trl进行Lora微调
GPU: Rtx 5090 32GB
模型：Qwen3-8B
"""
import os

from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from peft import LoraConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"]="./logs/Qwen3-8B-sft-lora-r-8"
model_name = 'model/Qwen3-8B'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


dataset_dict = load_dataset("json",data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})


def map_func(example):

    conversation = example["conversation"]
    messages = []
    for turn in conversation:
        messages.append({"role":"user","content":turn["human"]})
        messages.append({"role":"assistant","content":turn["assistant"]})
    
    return {"messages":messages}

dataset_dict = dataset_dict.map(function=map_func,batched=False,remove_columns=["dataset","conversation","category","conversation_id"])


peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)
training_args = SFTConfig(
    output_dir="./finetuned/Qwen3-8B-sft-lora",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=5e-5, # 这个指定的，是最大学习率吗？怎么调度的
    logging_steps=400,
    warmup_steps=0.1,
    eval_strategy="steps",
    eval_steps=400,
    load_best_model_at_end=True,
    bf16=True,
    save_total_limit=2,
    save_steps=400,  
    report_to=["tensorboard"]
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer,
    peft_config=peft_config
)

trainer.train()

trainer.save_model('./finetuned/Qwen3-8B-sft-lora-best')
