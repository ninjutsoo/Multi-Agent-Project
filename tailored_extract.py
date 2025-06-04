import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define constants
CSV_PATH = "/home/hq6375/Desktop/Code/Multi-Agent-Project/batch_extractions_only.csv"
OUTPUT_CSV = "/home/hq6375/Desktop/Code/Multi-Agent-Project/batch_extractions_with_tailored.csv"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

hft = os.getenv("HF_TOKEN")
if not hft:
    raise ValueError("❌ HF_TOKEN not found in environment variables! Make sure your .env file exists.")
MAX_TOKENS = 4096

# Load tokenizer and model with token and trust settings
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hft, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, token=hft, device_map="auto", torch_dtype=torch.float16)
model.eval()

def extract_information_tailored(note, hadm_id="UNKNOWN", label=0):
    tokenized_note = tokenizer(note, truncation=False, return_tensors="pt")
    note_tokens = len(tokenized_note["input_ids"][0])
    if note_tokens > MAX_TOKENS:
        print(f"⚠ Skipping HADM_ID {hadm_id} (Tokens: {note_tokens})")
        return None

    if label == 1:
        instruction = """Extract key clinical details that indicate the patient is at high risk for being readmitted within 30 days. Focus on critical diagnoses, unstable discharge conditions, high-risk medications, and urgent follow-up instructions. Your response must be in **valid JSON** format, following this exact structure:

{
    "past_hospitalizations": ["List prior hospital admissions with emphasis on chronic or recurring issues."],
    "discharge_diagnoses": ["List diagnoses suggesting high risk or ongoing instability."],
    "high_risk_medications": ["Include any medications associated with adverse outcomes or close monitoring (e.g., anticoagulants, insulin)."],
    "follow_up_instructions": ["Include follow-ups that imply ongoing risk or complexity (e.g., urgent follow-ups, multiple specialties)."],
    "discharge_disposition": "Where the patient is being sent (e.g., 'home with nursing', 'SNF', 'rehab').",
    "condition_at_discharge": "Summarize if patient is 'still symptomatic', 'fragile', or similar."
}

ONLY output the JSON. Do NOT include explanations, notes, or filler text. Be concise.
<|end|>
"""
    else:
        instruction = """Extract concise details that support patient stability and suggest **low risk** for 30-day readmission. Highlight absence of complex issues, stable conditions, and normal follow-up plans. Your response must be in **valid JSON** format, following this exact structure:

{
    "past_hospitalizations": ["List previous admissions if any, otherwise return []"],
    "discharge_diagnoses": ["List only non-critical or resolved diagnoses, otherwise return []"],
    "high_risk_medications": ["Only include if high-risk meds are present, else return []"],
    "follow_up_instructions": ["Include follow-ups that are routine or non-urgent, else return []"],
    "discharge_disposition": "Summarize simply, such as 'home' or 'home without services'.",
    "condition_at_discharge": "Summarize as 'stable', 'improved', or similar."
}

ONLY output the JSON. Do NOT include explanations, notes, or filler text. Be concise.
<|end|>
"""

    input_text = f"{instruction}\n\n{note}"
    tokenized_input = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(model.device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **tokenized_input,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
        extracted_info = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        first_brace = extracted_info.find("{")
        second_brace = extracted_info.find("{", first_brace + 1)
        json_end = extracted_info.rfind("}")
        json_start = second_brace if second_brace != -1 else first_brace
        extracted_json = extracted_info[json_start:json_end + 1].strip() \
            if json_start != -1 and json_end != -1 else "ERROR"

        print(f"✅ Extracted for HADM_ID {hadm_id}\n{extracted_json}\n{'-'*60}")
        return extracted_json
    except Exception as e:
        print(f"❌ Error for HADM_ID {hadm_id}: {e}")
        return "ERROR"

# Read CSV and generate tailored instructions
df = pd.read_csv(CSV_PATH)
tailored_outputs = []

for idx, row in df.iterrows():
    hadm_id = row.get("ID", f"row{idx}")
    note = row["Free Text"]
    label = int(row["Label"])
    result = extract_information_tailored(note, hadm_id, label)
    tailored_outputs.append(result)

df["Extracted_informed"] = tailored_outputs
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Done. Saved output to {OUTPUT_CSV}")
