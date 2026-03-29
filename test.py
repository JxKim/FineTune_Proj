def test_eos_token_id():
    from transformers import AutoTokenizer,AutoModelForCausalLM

    model_name = "model/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print('模型的eos_token_id为：',model.generation_config.eos_token_id)
    print('tokenizer的eos_token_id为：',tokenizer.eos_token_id)
    res = tokenizer.convert_ids_to_tokens([tokenizer.eos_token_id])
    text = tokenizer.apply_chat_template([{"role":"user","content":"你好"}],tokenize=False)
    print('tokenizer的eos_token_id对应的token为：',res[0])
    print('tokenizer的apply_chat_template对应的文本为：',text)

test_eos_token_id()