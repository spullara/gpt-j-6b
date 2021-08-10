from transformers import GPTNeoForCausalLM, AutoConfig, GPT2Tokenizer
import transformers

model = GPTNeoForCausalLM.from_pretrained("./gpt-j-hf")
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model.half().cuda()
input_text = "Hello my name is Paul and"
input_ids = tokenizer.encode(str(input_text), return_tensors='pt').cuda()

output = model.generate(
    input_ids,
    do_sample=True,
    max_length=20,
    top_p=0.7,
    top_k=0,
    temperature=1.0,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))

def eval(text):
    input_ids = tokenizer.encode(str(text), return_tensors='pt').cuda()
    output = model.generate(
        input_ids,
        do_sample=True,
        max_length=20,
        top_p=0.7,
        top_k=0,
        temperature=1.0,
    )

    print(tokenizer.decode(output[0], skip_special_tokens=True))
