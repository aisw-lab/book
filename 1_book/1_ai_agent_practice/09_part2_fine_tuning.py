import os
import torch
import csv
from datasets import load_dataset


# How to check HF access token: hugging face -> select account at top right -> settings -> access token
import huggingface_hub
huggingface_hub.login()

os.environ["NCCL_DEBUG"] = "INFO"
os.environ['TORCH_DISTRIBUTED_DEBUG']= "INFO"
os.environ['TORCH_SHOW_CPP_STACKTRACES']= "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use

torch.cuda.empty_cache()        # CUDA 메모리 정리

# GPU 할당 변경하기
GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

print('Device:', device)
print('Count of using GPUs:', torch.cuda.device_count())

torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

from huggingface_hub.hf_api import HfFolder
#HfFolder.save_token('토큰')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    Trainer, 
)
from peft import LoraConfig
from trl import SFTTrainer

# set base model path
#base_model = "beomi/Llama-3-Open-Ko-8B"
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

# set new model path
new_model = "Llama3-Ko-8b-meta"

dataset_namedodo = "사용자아이디/llama3data"
datasetdodo = load_dataset(dataset_namedodo, split="train")

def create_text_column(example):
    # 'text' 컬럼 생성
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    example["text"] = text
    return example

# 'text' 컬럼 생성
datasetdodo = datasetdodo.map(create_text_column, batch_size=6)

print(datasetdodo)

# 2. Config efficient fine-tuning with low-rank adaptation.

# 현재 사용 중인 GPU의 주요 아키텍처 버전을 반환 8버전 이상 시 bfloat16 활용
if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# BitsAndBytesConfig 객체활용 양자화 설정
# 모델을 4비트 양자화하여 로드
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=False,
)

# 3. Load Pre-trained Language Model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
    # device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 4. Load Pre-trained Language Model Tokenizer
# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# Must add EOS_TOKEN at response last line
tokenizer.pad_token = tokenizer.eos_token
# ★수정 포인트!!! 기존 # tokenizer.padding_side = "right"
EOS_TOKEN = tokenizer.eos_token
def prompt_eos(sample):
    sample['text'] = sample['text']+EOS_TOKEN
    return sample
datasetdodo = datasetdodo.map(prompt_eos)

print(datasetdodo[1])

# 5. Config training parameter for LoRA (Parameter-Efficient Fine-Tuning (PEFT)
#https://huggingface.co/docs/peft/conceptual_guides/lora

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

#per_device_train_batch_size=4에서 1로 변경(GPU MEM 문제)
#fp16=False로 변경
#bf16=False, 에서 주석 (Ampere GPU 미지원)
#gradient_checkpointing=True 추가

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,         # 배치 크기 4(4 * 1)
    gradient_checkpointing=True, 
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,                          # 부동소수점 변경(FP16 - Tesla M40 미지원)
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
)

# 6. Train Model
trainer = SFTTrainer(
    model=model,
    train_dataset=datasetdodo,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)


trainer.train()

# 7. Verify
def generate_response(prompt, model):
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')

    generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    # generated_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=True, pad_token_id=128009)

    decoded_output = tokenizer.batch_decode(generated_ids)

    return decoded_output[0].replace(prompt, "")

key = "컴퓨터공학을 전공하면 무슨 일을 하나요?"
prompt=f"""you are a assistant please answer in korean lanauage

### Instruction:
{key}

### Response:"""
generate_response(prompt, model)

#텍스트 생성을 위한 pipeline 생성 (모델 이름은 실제 모델로 변경 필요)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)  # GPU 사용

key = "컴퓨터공학을 전공하면 무슨 일을 하나요?"
prompt = f"""you are a assistant please answer in korean lanauage

### Instruction:
{key}

### Response:"""

response = generator(prompt, max_length=200, do_sample=True, pad_token_id=tokenizer.eos_token_id)

# 결과 출력
print(response[0]['generated_text'].replace(prompt, ""))

savePath = "/home/pc/1_dev/240623_llama_3/1_user_define_model/fine_tuning/llama3_meta_dodocode_0602"
trainer.save_model(savePath)