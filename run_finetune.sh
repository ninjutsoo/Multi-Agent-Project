#!/bin/bash

# === PATHS ===
TRAIN_DIR="/home/hq6375/Desktop/Code/Multi-Agent-Project/split_json_unbalanced"
SCRIPT_PATH="/home/hq6375/Desktop/Code/Multi-Agent-Project/train3.py"
LOSSES_DIR="/home/hq6375/Desktop/Code/Multi-Agent-Project"

# === MODEL SETUP ===
MODELS=("meta-llama/Meta-Llama-3-8B" "emilyalsentzer/Bio_ClinicalBERT")

# === SEEDS AND SHOT SIZES ===
SEEDS=(0 21 42 1337 1024)
SHOTS_LIST=(2 4 8 16 32 64 128)

# === ADAPTIVE HYPERPARAMETERS CONFIGURATION ===
# Fixed learning rate with shot-specific epoch scaling
BASE_LEARNING_RATE=1e-5      # Primary LR for most shot sizes
BASE_EPOCHS=40               # Base epochs, will be scaled per shot size
WEIGHT_DECAY=0.01        
MAX_GRAD_NORM=1.0        
LABEL_SMOOTHING=0.1      
WARMUP_STEPS=8           

echo "🗂️  Initializing dual-model adaptive few-shot training..."
echo "📋 Models to train: ${MODELS[*]}"
echo "📋 Shot-specific configuration:"
echo "   ✅ 2,4-shot: 1e-5 LR, 40 epochs"
echo "   ✅ 8-shot: 1e-6 LR, 60 epochs" 
echo "   ✅ 16,32-shot: 1e-5 LR, 80 epochs"
echo "   ✅ 64-shot: 1e-5 LR, 160 epochs"
echo "   ✅ 128-shot: 1e-5 LR, 320 epochs"
echo "   ✅ Dynamic dropout regularization and early stopping"
echo ""

# === DUAL MODEL TRAINING LOOP ===
for BASE_MODEL in "${MODELS[@]}"; do
    echo "🚀 ================================================="
    echo "🚀 STARTING TRAINING FOR MODEL: $BASE_MODEL"
    echo "🚀 ================================================="
    
    MODEL_NAME_CLEAN=$(echo "$BASE_MODEL" | tr '/:' '_')
    
    IS_BERT=false
    if [[ "${BASE_MODEL,,}" == *"bert"* ]]; then
        IS_BERT=true
    fi
    
    echo "🔬 Model type: $([ "$IS_BERT" = true ] && echo "BERT" || echo "LLaMA")"
    echo ""

    for SHOTS in "${SHOTS_LIST[@]}"; do
        echo "🎯 Starting ${SHOTS}-shot adaptive training for $BASE_MODEL across all seeds..."
        
        FIRST_SEED_FOR_SHOTS=true
        for SEED in "${SEEDS[@]}"; do
            TRAIN_FILE="$TRAIN_DIR/train_${SHOTS}_seed${SEED}.json"

            if [ ! -f "$TRAIN_FILE" ]; then
                echo "⚠️  Skipping: $TRAIN_FILE not found"
                continue
            fi

            # Adaptive batch sizes based on dataset size
            if [ "$SHOTS" -eq 2 ]; then
                BATCH_SIZE=2
                MICRO_BATCH=1
            elif [ "$SHOTS" -eq 4 ]; then
                BATCH_SIZE=4  
                MICRO_BATCH=2
            elif [ "$SHOTS" -le 16 ]; then
                BATCH_SIZE=8
                MICRO_BATCH=2
            else
                BATCH_SIZE=16
                MICRO_BATCH=4
            fi

            OUTPUT_DIR="/data/Amin/Models/${MODEL_NAME_CLEAN}/shots_${SHOTS}_seed${SEED}"
            mkdir -p "$OUTPUT_DIR"

            if [ "$FIRST_SEED_FOR_SHOTS" = true ]; then
                echo "🔍 Training ${SHOTS}-shot with adaptive hyperparameters..."
                echo "   📊 Base LR: ${BASE_LEARNING_RATE} (will be auto-scaled)"
                echo "   📊 Base epochs: ${BASE_EPOCHS} (will be auto-scaled)"
                echo "   📊 Regularization: weight_decay=${WEIGHT_DECAY}, label_smoothing=${LABEL_SMOOTHING}"
                FIRST_SEED_FOR_SHOTS=false
            fi
            
            echo "🔁 Training $SHOTS-shot, seed $SEED (adaptive hyperparameters)..."
            echo "📄 File: $TRAIN_FILE"
            echo "⚙️  batch_size=$BATCH_SIZE, micro_batch=$MICRO_BATCH"
            echo "📦 Output: $OUTPUT_DIR"
            echo "🔬 Model: $BASE_MODEL (BERT: $IS_BERT)"

            python "$SCRIPT_PATH" \
                --base_model "$BASE_MODEL" \
                --COT_train_file "$TRAIN_FILE" \
                --COT_dev_file "$TRAIN_DIR/validation.json" \
                --output_dir "$OUTPUT_DIR" \
                --batch_size $BATCH_SIZE \
                --micro_batch_size $MICRO_BATCH \
                --num_epochs $BASE_EPOCHS \
                --learning_rate $BASE_LEARNING_RATE \
                --weight_decay $WEIGHT_DECAY \
                --max_grad_norm $MAX_GRAD_NORM \
                --label_smoothing_factor $LABEL_SMOOTHING \
                --hft "$(python3 -c 'import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv(\"HF_TOKEN\"))')"

            echo "✅ Finished shots=$SHOTS seed=$SEED for $BASE_MODEL"
            echo "📋 Check training_issues.txt for any remaining critical issues"
            echo "----------------------------------------"
        done
        
        # Finalize reports for this shot size
        echo "🎯 Finalizing ${SHOTS}-shot analysis for $BASE_MODEL..."
        python3 -c "
import sys
import os
sys.path.insert(0, '/home/hq6375/Desktop/Code/Multi-Agent-Project')
os.chdir('/home/hq6375/Desktop/Code/Multi-Agent-Project')

try:
    from train3 import ShotReportManager
    manager = ShotReportManager()
    manager.finalize_shot_reports(${SHOTS})
    print('✅ Report finalization completed successfully')
except Exception as e:
    print(f'❌ Error finalizing reports: {e}')
    import traceback
    traceback.print_exc()
"
        echo "✅ ${SHOTS}-shot analysis complete for $BASE_MODEL"
        echo ""
    done
    
    echo "🏁 COMPLETED ALL TRAINING FOR MODEL: $BASE_MODEL"
    echo "📦 Models saved to: /data/Amin/Models/${MODEL_NAME_CLEAN}/shots_*"
    echo ""
done

echo "🎉 ================================================="
echo "🎉 ALL DUAL-MODEL TRAINING COMPLETED!"
echo "🎉 ================================================="
echo ""
echo "📊 TRAINING SUMMARY:"
echo "   🔬 Models trained: ${MODELS[*]}"
echo "   🎯 Shot sizes: ${SHOTS_LIST[*]}"
echo "   🎲 Seeds per shot: ${SEEDS[*]}"
echo "   📁 Total experiments: $((${#MODELS[@]} * ${#SHOTS_LIST[@]} * ${#SEEDS[@]}))"
echo ""
echo "📊 EXPECTED IMPROVEMENTS:"
echo "   🎯 Reduced overfitting through adaptive learning rates"
echo "   🎯 Better generalization with dynamic regularization"
echo "   🎯 Optimal training duration via adaptive epochs"
echo "   🎯 Early stopping to prevent overtraining"
echo ""
echo "📈 REPORTS: Check training_progress.txt for comparison across models"
echo "🐛 ISSUES: Check training_issues.txt - should see fewer critical issues now"
echo "📦 MODEL LOCATIONS:"
for model in "${MODELS[@]}"; do
    clean_name=$(echo "$model" | tr '/:' '_')
    echo "   ${model}: /data/Amin/Models/${clean_name}/"
done
