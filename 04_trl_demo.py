import os

from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
model_name = 'model/Qwen3-0.6B'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 配置tensorboard日志目录，还需要设置一个软链，因为在AutoDL中，Tensorboard默认读取的目录为/root/tf-logs
# 软链配置命令(注意，前面的路径/root/autodl-tmp/sft-train/logs需要根据个人项目命令来调整)：ln -s /root/autodl-tmp/sft-train/logs /root/tf-logs
os.environ["TENSORBOARD_LOGGING_DIR"]="./logs/Qwen3-0.6B-TRL-SFT-full"
dataset_dict = load_dataset("json",data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})


def map_func(example):
    """
    函数，将原始数据转换为模型输入格式，这里的限制主要是来自于SFTTrainer，它仅接收两种类型的输入
    Args:
        example:
    """
    conversation = example["conversation"]
    messages = []
    for turn in conversation:
        messages.append({"role":"user","content":turn["human"]})
        messages.append({"role":"assistant","content":turn["assistant"]})
    
    return {"messages":messages}

dataset_dict = dataset_dict.map(function=map_func,batched=False,remove_columns=["dataset","conversation","category","conversation_id"])

training_args = SFTConfig(
    output_dir="./finetuned/Qwen3-0.6B-TRL-SFT-full",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    gradient_accumulation_steps=3,
    learning_rate=5e-5, # 
    logging_steps=100,
    warmup_steps=0.1,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    bf16=True,
    save_total_limit=2,
    skip_memory_metrics=False,
    save_steps=100,
    report_to=["tensorboard"]
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer
)

trainer.train()

trainer.save_model('./finetuned/Qwen3-0.6B-TRL-SFT-full-best')
