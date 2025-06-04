import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration

# os.environ["HF_HOME"] = "/home/hq6375/huggingface_Amin"
# cache_dir = os.environ["HF_HOME"]

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from datetime import datetime

summary_file_path = '/home/hq6375/Desktop/Code/Multi-Agent-Project/summary.txt'
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate readmission model.')
    parser.add_argument('--num_shots', type=str, default="16", help='Number of training shots used.')
    parser.add_argument('--serialization', type=str, default="=", help='Number of training shots used.')
    parser.add_argument('--seed', type=str, default="=", help='Number of training shots used.')
    parser.add_argument('--model_path', type=str, default="/data/Amin/Models/bert-test-4", help='Path to the trained model checkpoint.')
    parser.add_argument('--test_file', type=str, default="/home/hq6375/Desktop/Code/Multi-Agent-Project/split_json_unbalanced/test.json", help='Path to the test data JSON.')
    parser.add_argument('--output_file', type=str, default="/home/hq6375/Desktop/Code/Multi-Agent-Project/results/emilyalsentzer_Bio_ClinicalBERT", help='Path to save the output CSV.')


    args = parser.parse_args()
    return args
args = parse_arguments()


yes_probs = []
actual_labels = []

train_file_address = '/home/hq6375/Desktop/Code/Multi-Agent-Project/split_json/test.json'
# test_file_address = '/home/hq6375/Desktop/Code/Multi-Agent-Project/split_json/test.json'
test_file_address = args.test_file


# test_file_address = '/media/sdb/amin/Dropbox/code/NIH/Dataset-'+args.serialization +'/test_data_seed'+args.seed+'.json'

# Load training and testing data from JSON files
with open(train_file_address, 'r') as file:
    training_data = json.load(file)

with open(test_file_address, 'r') as file:
    all_testing_data = json.load(file)

# Configurable number of test and training samples
num_tests = 1000  # Adjust this to change the number of test samples
num_train = 0  # Adjust this to change the total number of training samples

# Ensure an equal distribution of severe and non-severe cases in training
num_severe = num_non_severe = num_train // 2

# Select the required number of test samples
testing_data = all_testing_data

# Setup for tokenizer and model
device = "cuda:0"
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("❌ HF_TOKEN not found in environment variables! Make sure your .env file exists.")
# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf", token=token)
# Load tokenizer from base model instead of fine-tuned checkpoint
if "bert" in args.model_path.lower():
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
else:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=token)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "right"  # BERT-style


num_shots = args.num_shots
# model_name = f"llama2-7b-{num_shots}shot"
# model_name = "llama2-7b-256shotfsfsfs"

# tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp", token=token)
# model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-13b-hf", token=token, load_in_8bit=True, torch_dtype=torch.float16)
# model = AutoModelForCausalLM.from_pretrained("/media/sdb/amin/Dropbox/code/NIH/", token=token, load_in_8bit=True, torch_dtype=torch.float16)

# model = AutoModelForCausalLM.from_pretrained("/media/sdb/amin/Dropbox/code/NIH/", token=token, torch_dtype=torch.float16).to(device)
# model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b", token=token, torch_dtype=torch.float16).to(device)

# model = AutoModelForCausalLM.from_pretrained("/media/sdb/amin/Dropbox/code/NIH/", token=token, torch_dtype=torch.float16).to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained("/media/sdb/amin/Dropbox/code/NIH//flan-t5-XL", token=token, torch_dtype=torch.float16, load_in_8bit=True)
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-XL", token=token, torch_dtype=torch.float16).to(device)

# model = AutoModelForCausalLM.from_pretrained("/media/sdb/amin/Dropbox/code/NIH/Models/llama2-7b-covid-8shot", token=token, torch_dtype=torch.float16).to(device)
# model = AutoModelForCausalLM.from_pretrained("/data/Amin/Models/llama3.1-8B-yes-no/checkpoint-100", token=token, torch_dtype=torch.float16).to(device)
# model = AutoModelForCausalLM.from_pretrained(args.model_path, token=token, torch_dtype=torch.float16).to(device)



# model = AutoModelForSeq2SeqLM.from_pretrained("NousResearch/Llama-2-7b-hf", token=token, torch_dtype=torch.float16).to(device)

# model = AutoModelForSeq2SeqLM.from_pretrained("/media/sdb/amin/Dropbox/code/NIH//T0_3B-heart", token=token, torch_dtype=torch.float16).to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained("/media/sdb/amin/Dropbox/code/NIH//T0pp", token=token, load_in_8bit=True, torch_dtype=torch.float16)

if "flan" in args.model_path or "T0" in args.model_path:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True
    ).to(device)
elif "bert" in args.model_path.lower():
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2,
    ).to(device)
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
    token=token,
    torch_dtype=torch.bfloat16,
    ).to(device)

print(tokenizer.name_or_path)
print(model.name_or_path)




# Function to format questions with alternating severity
instruction = "Based on the patient information above at discharge, will they be readmitted in 30 days? yes or no?\nResult: "
# instruction = "Does the coronary angiography of this patient show a heart disease? yes or no?"

def format_training_questions(training_data, num_severe, num_non_severe):
    
    # Process data
    severe_data = [item for item in training_data if item['label'] == 'yes']
    non_severe_data = [item for item in training_data if item['label'] == 'no']
    random.shuffle(severe_data)
    random.shuffle(non_severe_data)
    severe_data = severe_data[:num_severe]
    non_severe_data = non_severe_data[:num_non_severe]

    combined_data = []
    for s, ns in zip(severe_data, non_severe_data):
        combined_data.extend([ns, s])

    # Add remaining items if lists are of unequal length
    combined_data.extend(severe_data[len(non_severe_data):] + non_severe_data[len(severe_data):])

    # Format questions
    # questions = [instruction]
    questions = []

    for item in combined_data:
        question = f"\n{item['text']} Result:{item['label']}"
        questions.append(question)
    return questions

def format_test_questions(test_data, num_tests):
    questions = []
    answers = []

    # Ensure num_tests does not exceed the length of test_data
    num_tests = min(num_tests, len(test_data))

    # Randomly select num_tests items from test_data
    selected_items = random.sample(test_data, num_tests)

    for item in selected_items:
        # question = f"\nSample:{item['question']} Result:"
        # question = f"\n{item['question']} Result:"
        ### for zero-shot
        # question = f"Below you see the information about a patient where the importance of the information is in increasing order, therefore make sure to focus on the ending features more.\n{item['question']}\n{instruction} Result:"
        question = f"{item['text']}\n{instruction}"


        questions.append(question)
        answers.append(item['label'])

    return questions, answers

# Generate training and test questions
training_questions = format_training_questions(training_data, num_severe, num_non_severe)
test_questions, answer = format_test_questions(testing_data, num_tests)



# def predict_severity(model_input, tokenizer, model, device):
#     # Encode the input text
#     # encoding = tokenizer.encode_plus(model_input, return_tensors='pt', padding=True, truncation=True).to(device)
#     # encoding = tokenizer.encode_plus(model_input, return_tensors='pt', padding=True, truncation=True)
#     # model_input = "the opposite of good is "
#     encoding = tokenizer(model_input, return_tensors='pt', padding=True, truncation=True).to(device)
#     print(encoding['input_ids'].shape)

#     # output = model.generate(**encoding, max_new_tokens=2)
#     # print(tokenizer.decode(output[0]))
#     # # encoding = tokenizer(model_input, return_tensors='pt', padding=True, truncation=True).to(device)

#     input_ids = encoding['input_ids'].to(device)
    
#     model.eval()  # Ensure the model is in evaluation mode
    
#     with torch.no_grad():  # Disable gradient calculation for inference
#         if isinstance(model, T5ForConditionalGeneration):
#             sos_token_id = tokenizer.pad_token_id
#             decoder_input_ids = torch.full((input_ids.shape[0], 1), sos_token_id, dtype=torch.long).to(device)
#             outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
#             logits = outputs.logits
#         else:
#             outputs = model(input_ids=input_ids)
#             logits = outputs.logits
#     print(logits.shape)

#     # Extract logits for the top 5 probabilities and their indices (token IDs)

#     top_5_probs, top_5_token_ids = torch.topk(F.softmax(logits[:, -1, :], dim=1), 5, dim=1)
#     print(tokenizer.decode(top_5_token_ids[0]))
#     top_5_tokens_texts = tokenizer.convert_ids_to_tokens(top_5_token_ids.squeeze().tolist())
#     top_5_probs_percentage = top_5_probs.squeeze().tolist()

#     # Initialize probabilities for "yes" and "no"
#     yes_prob = no_prob = None
#     yes_found = False
#     no_found = False
#     # Iterate over the top 5 tokens to find "yes" or "no" and their probabilities
#     for token, prob in zip(top_5_tokens_texts, top_5_probs_percentage):
#         if "yes" in token.lower() and not yes_found:
#             yes_found = True
#             yes_prob = prob
#         elif "no" in token.lower() and not no_found:
#             no_found = True
#             no_prob = prob

#     # Check if either "yes" or "no" was not found and raise an error
#     if yes_found is False and no_found is False:
#          raise ValueError("Could not find probabilities for both 'yes' and 'no' in the top 5 tokens.")

#     if yes_found == False:
#         yes_prob = 1 - no_prob
#     if no_found == False:
#         no_prob = 1 - yes_prob

#     # Scale the probabilities to sum up to 100%
#     total_prob = yes_prob + no_prob
#     yes_prob_scaled = yes_prob / total_prob
#     no_prob_scaled = no_prob / total_prob

#     # Generate the output to decide on "severe" or "non-severe"
#     output_ids = model.generate(input_ids, max_length=input_ids.shape[1] + 1)
#     last_token_id = output_ids[0][-1]
#     last_token = tokenizer.decode([last_token_id])

#     # Determine the prediction based on the last token
#     prediction = "yes" if yes_prob_scaled>=no_prob_scaled else "no"
    
#     return prediction, last_token, top_5_tokens_texts, top_5_probs_percentage, yes_prob_scaled

yes_token = "yes"
no_token = "no"

def predict_severity(model_input, tokenizer, model, device):
    # Encode the input text
    encoding = tokenizer(model_input, return_tensors='pt', padding=True, truncation=True).to(device)
    print(encoding['input_ids'].shape)
    input_ids = encoding['input_ids'].to(device)

    model.eval()  # Ensure the model is in evaluation mode

    with torch.no_grad():
        # Step 1: Generate "Result:" token
        result_ids = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False)
        
        # Step 2: Append the generated "Result:" token to the input
        # extended_input_ids = torch.cat([input_ids, result_ids[:, -1:]], dim=1)

        # Step 3: Feed this into the model to get logits for the token after "Result:"
        outputs = model(**encoding)
        logits = outputs.logits
    print(logits.shape)

    # Extract top 5 logits for the token after "Result:"
    top_5_probs, top_5_token_ids = torch.topk(F.softmax(logits[:, -1, :], dim=1), 5, dim=1)
    print(tokenizer.decode(top_5_token_ids[0]))
    top_5_tokens_texts = [tokenizer.decode([token_id]) for token_id in top_5_token_ids[0]]
    top_5_probs_percentage = top_5_probs.squeeze().tolist()

    # Initialize probabilities for "yes" and "no"
    yes_prob = no_prob = None
    yes_found = False
    no_found = False

    for token, prob in zip(top_5_tokens_texts, top_5_probs_percentage):
        if yes_token in token.lower() and not yes_found:
            yes_found = True
            yes_prob = prob
        elif no_token in token.lower() and not no_found:
            no_found = True
            no_prob = prob

    if not yes_found and not no_found:
        raise ValueError("Could not find probabilities for both 'yes' and 'no' in the top 5 tokens.")
    if not yes_found:
        yes_prob = 1 - no_prob
    if not no_found:
        no_prob = 1 - yes_prob

    # Scale the probabilities to sum to 1
    total_prob = yes_prob + no_prob
    yes_prob_scaled = yes_prob / total_prob
    no_prob_scaled = no_prob / total_prob

    # Step 4: Fully generate "Result: yes" or "Result: no"
    full_output_ids = model.generate(input_ids=input_ids, max_new_tokens=2)
    last_token_id = full_output_ids[0][-1]
    last_token = tokenizer.decode([last_token_id])

    # Final prediction based on probability
    prediction = yes_token if yes_prob_scaled >= no_prob_scaled else no_token

    return prediction, last_token, top_5_tokens_texts, top_5_probs_percentage, yes_prob_scaled



# Run predictions for test data
results = []
correct_predictions = 0  # Initialize the count of correct predictions

TP = FP = TN = FN = 0

for i, test_question in enumerate(test_questions):
    # model_input = "".join(training_questions + [test_question])
    model_input = test_question if isinstance(test_question, str) else "".join(test_question)

    if "bert" in tokenizer.name_or_path.lower():
        # For BERT: classification prediction using logits
        encoding = tokenizer(
            model_input,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoding)
            # print(tokenizer.decode(outputs[0]))
            logits = outputs.logits
            print(logits.shape)
            probs = torch.softmax(logits, dim=-1).squeeze()
            yes_prob = probs[1].item()  # probability of "yes" class
            predicted_severity = "yes" if yes_prob >= 0.5 else "no"
            last_token = "[CLS]"
            top_5_tokens_texts = ["no", "yes"]  # reflect actual label order
            top_5_probs_percentage = [probs[0].item(), probs[1].item()]  # [P(no), P(yes)]

    else:
        # For LLaMA/Generative: use original generation logic
        encoding = tokenizer(model_input, return_tensors='pt', padding=True, truncation=True).to(device)
        output = model.generate(**encoding, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(output[0][-1:]))
        print(output[0][-1:])
        print(answer[i])
        print('--' * 50)

        predicted_severity, last_token, top_5_tokens_texts, top_5_probs_percentage, yes_prob = predict_severity(
            model_input, tokenizer, model, device
        )

    actual_severity = answer[i]
    is_predicted_severe = yes_token in predicted_severity.lower()
    is_actual_severe = "yes" in actual_severity.lower()

    yes_probs.append(yes_prob)  # Store the probability for "yes"
    actual_labels.append(1 if actual_severity == 'yes' else 0)

    # Update confusion matrix counts
    if is_predicted_severe and is_actual_severe:
        TP += 1
    elif is_predicted_severe and not is_actual_severe:
        FP += 1
    elif not is_predicted_severe and not is_actual_severe:
        TN += 1
    elif not is_predicted_severe and is_actual_severe:
        FN += 1

    print(model_input)
    print("actual: ", actual_severity)
    print("predicted: ", "yes" if is_predicted_severe else "no")
    if "bert" in tokenizer.name_or_path.lower():
        print(f"Logits: {logits.tolist()} → Softmax: {probs.tolist()}")
    else:
        print(top_5_tokens_texts, "  ", top_5_probs_percentage)

    print(i)

    # Append information to results
    results.append((i, actual_severity, top_5_tokens_texts[0], predicted_severity))


# Calculate accuracy from the confusion matrix
accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) else 0

# Print the confusion matrix and accuracy
print(f"Confusion Matrix and Accuracy:\nTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, Accuracy: {accuracy}")
print(f"Accuracy: {accuracy:.4f}")
auc_score = roc_auc_score(actual_labels, yes_probs)
print(f"AUC Score: {auc_score:.4f}")

##################################
df = pd.DataFrame({
    "Actual Labels": actual_labels,
    "Yes Probabilities": yes_probs
})

# csv_file_path = '/media/sdb/amin/Dropbox/code/NIH/ROC-'+args.serialization +'/' + model_name + "_seed"+args.seed+".csv"
# csv_file_path = "/home/hq6375/Desktop/Code/Multi-Agent-Project/result.csv"
csv_file_path = args.output_file


# Ensure the directory exists
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
# Save to CSV, this line creates the CSV if it does not exist
df.to_csv(csv_file_path, index=False)
print(f"File saved to {csv_file_path}")
##################################

# Save results and confusion matrix to CSV
# results_df = pd.DataFrame(results, columns=['Test Index', 'Real Severity', 'Top Prediction', 'Predicted Severity'])
confusion_matrix_df = pd.DataFrame([[TP, FP, TN, FN]], columns=['TP', 'FP', 'TN', 'FN'])
# results_csv_path = '/media/sdb/amin/Dropbox/code/NIH//10-s-equalstyle.csv'
# confusion_matrix_csv_path = '/media/sdb/amin/Dropbox/code/NIH/confusion_ma0trix.csv'
confusion_matrix_csv_path = '/home/hq6375/Desktop/Code/Multi-Agent-Project/confusion_matrix.csv'


# results_df.to_csv(results_csv_path, index=False)
confusion_matrix_df.to_csv(confusion_matrix_csv_path, index=False)

# print(f"Results saved to {results_csv_path}")
print(f"Confusion Matrix saved to {confusion_matrix_csv_path}")

summary_info = f"""{instruction}
training file: {csv_file_path}
tokenizer: {tokenizer.name_or_path}
model: {model.name_or_path}
num of training cases: {num_train}
TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Seed: {args.seed}

######################
"""
# if num_train != 0:
#     # Append the information to the summary file
#     with open(summary_file_path+".txt", 'a') as file:
#         file.write(summary_info)
# else:
#     with open(summary_file_path+"-finetuned-"+args.serialization +".txt", 'a') as file:
#         file.write(summary_info)

print(f"Summary information saved to {summary_file_path}")
