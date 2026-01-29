import pandas as pd
import torch
from transformers import pipeline, MarianMTModel, MarianTokenizer
from tqdm import tqdm

# def convert_icd_to_excel(input_txt: str, output_xlsx: str):
#     # Чтение текстового файла с кодами ICD-10
#     rows = []
    
#     with open(input_txt, encoding='utf-8') as f:
#         for line in f:
#             parts = line.strip().split(None, 1)
#             if len(parts) == 2:
#                 code, eng_desc = parts
#             elif len(parts) == 1:
#                 code, eng_desc = parts[0], ""
#             rows.append({'Код': code, 'Описание на русском': eng_desc})
            
#     df = pd.DataFrame(rows)
    
#     # Перевод описаний с помощью трансформеров
#     translator = pipeline("translation_en_to_ru", model="Helsinki-NLP/opusmt-en-ru")
    
#     df['Описание на русском'] = df['Описание на русском'].apply(
#         lambda text: translator(text, max_length=128)[0]['translation_text']
#     )
    
#     # Сохранение в Excel (движок openpyxl по умолчанию)
#     df.to_excel(output_xlsx, index=False)
    
    
def translate_icd10(input_txt: str, output_xlsx: str, batch_size: int = 16):
    rows = []

    with open(input_txt, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                code, text = parts
                rows.append({"Код": code, "Описание": text})

    df = pd.DataFrame(rows)

    model_name = "Helsinki-NLP/opus-mt-en-ru"

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    translated = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df["Описание"].iloc[i:i + batch_size].tolist()

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs)

        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(texts)

    df["Описание"] = translated
    df.to_excel(output_xlsx, index=False)
    
if __name__ == "__main__":
    translate_icd10(
        input_txt="data/icd10_en.txt",
        output_xlsx="data/icd10_ru.xlsx"
    )