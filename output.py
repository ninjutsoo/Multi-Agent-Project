#cell 1
import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Set Hugging Face Token
hft = os.getenv("HF_TOKEN")
if not hft:
    raise ValueError("❌ HF_TOKEN not found in environment variables! Make sure your .env file exists.")

# ✅ Set Hugging Face Cache Directory
os.environ["HF_HOME"] = "/home/hq6375/huggingface_Amin"
cache_dir = os.environ["HF_HOME"]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ✅ Define GPU Mode (0: GPU 0, 1: GPU 1, 2: Both GPUs)
GPU_MODE = 2  # Change this variable only (0, 1, or 2)

# ✅ Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("❌ No CUDA-compatible GPU found!")

# ✅ Select Device Mapping Based on GPU Mode
if GPU_MODE == 0:
    device_map = {"": 0}  # Load everything on GPU 0
    print(f"✅ Using GPU 0: {torch.cuda.get_device_name(0)}")
elif GPU_MODE == 1:
    device_map = {"": 1}  # Load everything on GPU 1
    print(f"✅ Using GPU 1: {torch.cuda.get_device_name(1)}")
elif GPU_MODE == 2:
    device_map = "auto"  # Automatically distribute across GPUs
    print(f"✅ Using Multiple GPUs: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_name(1)}")
else:
    raise ValueError("❌ Invalid GPU_MODE! Use 0 (GPU 0), 1 (GPU 1), or 2 (both GPUs).")

# ✅ Model ID
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


# ✅ Load tokenizer with authentication
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hft, trust_remote_code=True)

# ✅ Load model and distribute across GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hft,
    torch_dtype=torch.bfloat16,  # Use bf16 for lower memory usage
    low_cpu_mem_usage=True,  # Prevent CPU offloading
    trust_remote_code=True,
    device_map=device_map  # Automatically maps model to GPU(s)
)

print("✅ Mistral-7B-Instruct-v0.3 successfully loaded according to GPU_MODE!")


# ✅ CELL 2: Information Extraction from Note
MIN_TOKENS = 1000
MAX_TOKENS = 4000

def extract_information(note, hadm_id="UNKNOWN"):
    tokenized_note = tokenizer(note, truncation=False, return_tensors="pt")
    note_tokens = len(tokenized_note["input_ids"][0])
    if note_tokens > MAX_TOKENS:
        print(f"⚠ Skipping HADM_ID {hadm_id} (Tokens: {note_tokens})")
        return None

    instruction = """Extract only the most relevant medical details that may impact 30-day readmission risk from the following discharge summary. Your response must be in **valid JSON** format, short and precise, and follow this exact structure:

{
    "past_hospitalizations": ["List prior hospital admissions briefly, e.g., 'CHF in 2021', or return [] if none are mentioned."],
    "discharge_diagnoses": ["List only the main diagnoses at discharge, no explanations or duplication, or return [] if none are stated."],
    "high_risk_medications": ["List high-risk medications prescribed at discharge (e.g., insulin, opioids, anticoagulants), or return [] if none."],
    "follow_up_instructions": ["List specific follow-ups (e.g., 'Cardiology in 2 weeks'), or return [] if not clearly given."],
    "discharge_disposition": "Where the patient is being sent after discharge (e.g., 'home', 'SNF', 'rehab', 'another hospital'), or return \"\".",
    "condition_at_discharge": "Summarize discharge stability in a few words (e.g., 'stable', 'still symptomatic'), or return \"\"."
}

ONLY output the JSON. Do NOT include explanations, notes, or filler text. Be brief and structured.
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
        extracted_json = extracted_info[json_start:json_end + 1].strip() if json_start != -1 and json_end != -1 and "List prior hospital admissions briefly" not in extracted_info[json_start:json_end + 1].strip() else "ERROR"
        print(f"\n✅ HADM_ID {hadm_id} | Extracted Output:\n{extracted_json}\n{'-'*80}")
        return extracted_json
    except Exception as e:
        print(f"❌ Error extracting info from HADM_ID {hadm_id}: {e}")
        return "ERROR"

# ✅ CELL 3: Print function (unchanged)
def print_extracted_patient(index=0):
    if "extracted_results" not in globals() or len(extracted_results) == 0:
        print("⚠ No extracted results found. Run the extraction first.")
        return
    if index < 0 or index >= len(extracted_results):
        print(f"⚠ Invalid index {index}.")
        return
    patient = extracted_results[index]
    print("\n" + "="*80)
    print(f"🩺 Patient ID: {patient.get('patient_id', 'Unknown')}")
    print(f"📄 Note Preview:\n{patient.get('original_notes', '')[:200]}...")
    print(f"📝 Extracted JSON:\n{patient.get('extracted_info', '')}\n")
    print("="*80)

# ✅ CELL 4: Load discharge summaries only
import pandas as pd
note_path = "/home/hq6375/Desktop/Code/MIMIC-III/physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv"
admissions_path = "/home/hq6375/Desktop/Code/MIMIC-III/physionet.org/files/mimiciii/1.4/ADMISSIONS.csv"
cleaned_path = "/home/hq6375/Desktop/Code/MIMIC-III/cleaned_NOTES.csv"

if os.path.exists(cleaned_path):
    df = pd.read_csv(cleaned_path)
    print("✅ Loaded cleaned data.")
else:
    df = pd.read_csv(note_path)
    admissions = pd.read_csv(admissions_path)
    df = df[df["CATEGORY"] == "Discharge summary"]
    df = df.sort_values(by=["SUBJECT_ID", "HADM_ID", "CHARTDATE"])
    df = df.groupby(["SUBJECT_ID", "HADM_ID"]).tail(1)
    df = df.dropna(subset=["TEXT"])
    df["CLEANED_TEXT"] = df["TEXT"].str.replace(r"\n|\r", " ", regex=True).str.lower().str.strip()
    df.to_csv(cleaned_path, index=False)
    print("✅ Cleaned and saved notes.")
print(f"📊 Total notes: {len(df)}")

# ✅ CELL 5: Generate readmission labels
admissions = pd.read_csv(admissions_path)
admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"])
admissions["DEATHTIME"] = pd.to_datetime(admissions["DEATHTIME"])
admissions = admissions.sort_values(by=["SUBJECT_ID", "ADMITTIME"])
admissions["NEXT_ADMITTIME"] = admissions.groupby("SUBJECT_ID")["ADMITTIME"].shift(-1)
admissions["NEXT_ADMISSION_TYPE"] = admissions.groupby("SUBJECT_ID")["ADMISSION_TYPE"].shift(-1)
admissions.loc[admissions["NEXT_ADMISSION_TYPE"] == "ELECTIVE", "NEXT_ADMITTIME"] = pd.NaT
admissions = admissions[admissions["DEATHTIME"].isna()]
admissions = admissions[admissions["ADMISSION_TYPE"] != "NEWBORN"]
admissions["readmission_30d"] = ((admissions["NEXT_ADMITTIME"] - admissions["DISCHTIME"]).dt.total_seconds() < 30 * 86400).astype(int)
readmission_labels = admissions[["SUBJECT_ID", "HADM_ID", "readmission_30d"]]
readmission_labels.to_csv("readmission_labels.csv", index=False)
print("✅ Labels saved.")

# ✅ CELL 6: Merge labels into notes
df = df.merge(readmission_labels, on=["SUBJECT_ID", "HADM_ID"], how="inner")
print(f"✅ Notes with readmission labels: {len(df)}")

# ✅ CELL 7: Extraction-only mode (with correct ClinicalBERT-style labels)
import torch.nn.functional as F
import json

# ✅ Token estimation for filtering
df["CLEANED_TEXT"] = df["CLEANED_TEXT"].astype(str)
df["TOKEN_ESTIMATE"] = df["CLEANED_TEXT"].apply(lambda x: len(x) // 4)
filtered_df = df[(df["TOKEN_ESTIMATE"] >= MIN_TOKENS) & (df["TOKEN_ESTIMATE"] <= MAX_TOKENS)].copy()

# ✅ Output file path (extractions only)
save_path = "./batch_extractions_only.csv"
if not os.path.exists(save_path):
    pd.DataFrame(columns=[
        "HADM_ID", "CLEANED_TEXT", "extracted_json", "readmission_30d"
        # "predicted_token", "prob_yes", "prob_no", "prob_yes_normalized", "top_5_tokens"  # <- prediction columns commented
    ]).to_csv(save_path, index=False)

# ✅ Extraction loop config
results = []
max_samples = 300
attempts = 0
max_attempts = 1000

while len(results) < max_samples and attempts < max_attempts:
    if filtered_df.empty:
        print("❌ No more notes available.")
        break

    row = filtered_df.sample(1)
    hadm_id = row["HADM_ID"].values[0]
    text = row["CLEANED_TEXT"].values[0]
    label = row["readmission_30d"].values[0]  # ✅ ClinicalBERT-style label
    filtered_df = filtered_df[filtered_df["HADM_ID"] != hadm_id]

    extracted_json = extract_information(text, hadm_id=hadm_id)

    if not extracted_json or not extracted_json.startswith("{") or "List past hospitalizations" in extracted_json:
        print(f"⚠ Skipping HADM_ID {hadm_id} due to invalid extraction.")
        attempts += 1
        continue

    row_data = {
        "HADM_ID": hadm_id,
        "CLEANED_TEXT": text,
        "extracted_json": extracted_json,
        "readmission_30d": label,
        # "predicted_token": None,
        # "prob_yes": None,
        # "prob_no": None,
        # "prob_yes_normalized": None,
        # "top_5_tokens": None
    }

    # ✅ Save immediately to file
    pd.DataFrame([row_data]).to_csv(save_path, mode="a", header=False, index=False)
    results.append(row_data)

    print(f"✅ Saved {len(results)}/{max_samples} extractions (HADM_ID: {hadm_id})")
    attempts += 1

# ✅ Uncomment the block below to re-enable prediction
# instruction = (
#     "Based on the following patient data, will the patient be readmitted within 30 days?\n"
#     "You must answer using **only one token**, either 'yes' or 'no'.\n"
# )
# llm_input = f"{instruction}\n{extracted_json}\n\nAnswer:"
# tokenized_input = tokenizer(llm_input, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(model.device)

# with torch.no_grad():
#     logits = model(**tokenized_input).logits
#     last_logits = logits[0, -1]
#     probs = F.softmax(last_logits, dim=-1)
#     yes_token_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
#     no_token_id = tokenizer("no", add_special_tokens=False).input_ids[0]
#     prob_yes = probs[yes_token_id].item()
#     prob_no = probs[no_token_id].item()
#     prob_yes_norm = prob_yes / (prob_yes + prob_no)
#     topk = torch.topk(probs, k=5)
#     top_tokens = [(tokenizer.decode([i]).strip(), round(p.item(), 5)) for i, p in zip(topk.indices, topk.values)]

#     row_data.update({
#         "predicted_token": "yes" if prob_yes > prob_no else "no",
#         "prob_yes": prob_yes,
#         "prob_no": prob_no,
#         "prob_yes_normalized": prob_yes_norm,
#         "top_5_tokens": json.dumps(top_tokens)
#     })

# ✅ Final status message
if not results:
    print("❌ No valid extractions were completed.")
else:
    print(f"🎉 Done! Extracted and saved {len(results)} valid samples.")

