#!/bin/bash

# === MODEL BASE PATH ===
# MODEL_BASE="/data/Amin/Models/emilyalsentzer_Bio_ClinicalBERT"
MODEL_BASE="/data/Amin/Models/meta-llama_Meta-Llama-3-8B"

# === CONSTANT PATHS ===
TEST_FILE="/home/hq6375/Desktop/Code/Multi-Agent-Project/split_json_unbalanced/test.json"
SCRIPT_PATH="/home/hq6375/Desktop/Code/Multi-Agent-Project/test.py"
RESULTS_BASE="/home/hq6375/Desktop/Code/Multi-Agent-Project/results"

# === TAG MODEL NAME ===
MODEL_TAG=$(basename "$MODEL_BASE")
RESULT_DIR="${RESULTS_BASE}/${MODEL_TAG}"
mkdir -p "$RESULT_DIR"

# === DETERMINE MODEL TYPE ===
MODEL_NAME_LOWER=$(basename "$MODEL_BASE" | tr '[:upper:]' '[:lower:]')
IS_BERT=false
if [[ "$MODEL_NAME_LOWER" == *"bert"* ]]; then
    IS_BERT=true
fi

# === LOOP ONLY THROUGH SHOT-SEED FOLDERS ===
find "$MODEL_BASE" -maxdepth 1 -type d -name "shots_*_seed*" | sort -V | while read -r MODEL_DIR; do
    BASENAME=$(basename "$MODEL_DIR")
    
    # Extract shot count and seed
    SHOT=$(echo "$BASENAME" | grep -oP '(?<=shots_)\d+')
    SEED=$(echo "$BASENAME" | grep -oP '(?<=seed)\d+')
    
    OUTPUT_CSV="${RESULT_DIR}/shots_${SHOT}_seed${SEED}.csv"

    echo "🔍 Evaluating model: shots=$SHOT | seed=$SEED"
    echo "📂 Model path: $MODEL_DIR"
    echo "📄 Test file:  $TEST_FILE"
    echo "💾 Output CSV: $OUTPUT_CSV"

    # === CALL PYTHON SCRIPT BASED ON MODEL TYPE ===
    if [ "$IS_BERT" = true ]; then
        echo "🧠 Detected BERT-style classification model"
    else
        echo "💡 Detected autoregressive generative model"
    fi

    python "$SCRIPT_PATH" \
        --model_path "$MODEL_DIR" \
        --test_file "$TEST_FILE" \
        --output_file "$OUTPUT_CSV" \
        --num_shots "$SHOT" \
        --serialization "$MODEL_TAG" \
        --seed "$SEED"

    echo "✅ Completed shots=$SHOT seed=$SEED"
    echo "--------------------------------------"
done

echo "🎉 All evaluations completed and saved under: $RESULT_DIR"
