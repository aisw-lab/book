# How to check HF access token: hugging face -> select account at top right -> settings -> access token
import huggingface_hub

import pandas as pd
import json
import jsonlines
from datasets import Dataset
import os

REPO_NAME = "레포지토리/레포지토리명"
AUTH_TOKEN = "토큰값"
huggingface_hub.login(token=AUTH_TOKEN)

# Define paths and filenames
fileName = "indata_kor"
#dataset_Path = "D:/8_dev/2_python/python_lab/example/llama_repository/dataset"
dataset_Path = "/home/pc/1_dev/250105_llama_3/1_user_define_model/dataset"
csv_file_path = os.path.join(dataset_Path, f"{fileName}.csv")

# Load Data
try:
    df = pd.read_csv(csv_file_path, encoding="ms949")
except Exception as e:
    df = pd.read_csv(csv_file_path, encoding="utf-8")

# 빈컬럼 생성 및 컬럼 재정의
df["input"]=""
df.columns = ["instruction","output","input"]
df.head(2)

dataset = Dataset.from_pandas(df)

print(dataset)

# 필요시 허깅페이스에 업로드
dataset.push_to_hub("레포지토리/레포지토리명")

