"""LLM local (transformers). En Colab T4: 4 bits con bitsandbytes."""
import torch


class LocalLLM:
    def __init__(self, name="Qwen/Qwen2.5-7B-Instruct", load_4bit=True, max_new_tokens=256):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(name)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4") if load_4bit else None
        self.m = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=q, torch_dtype=torch.float16, device_map="cuda")
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate(self, messages):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to("cuda")
        out = self.m.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def free(self):
        del self.m
        torch.cuda.empty_cache()
