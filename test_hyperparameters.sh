#!/bin/bash

# === SYSTEMATIC HYPERPARAMETER TESTING ===
# Tests 3 LR scales (HIGH/MEDIUM/LOW) for each shot size
# LR scales differ by ~10x: 1e-5, 1e-6, 1e-7 effective ranges
# Adjusts epochs inversely: lower LR = more epochs

TRAIN_DIR="/home/hq6375/Desktop/Code/Multi-Agent-Project/split_json_unbalanced"
SCRIPT_PATH="/home/hq6375/Desktop/Code/Multi-Agent-Project/train3.py"
LOSSES_DIR="/home/hq6375/Desktop/Code/Multi-Agent-Project"
BASE_MODEL="emilyalsentzer/Bio_ClinicalBERT"
MODEL_NAME_CLEAN=$(echo "$BASE_MODEL" | tr '/:' '_')

# Fixed seed for consistency across tests
TEST_SEED=21

# Shot sizes to test
SHOTS_LIST=(2 4 8 16 32 64 128)

# Base hyperparameters
BASE_LR=2e-5
BASE_EPOCHS=20

echo "🧪 SYSTEMATIC HYPERPARAMETER TESTING"
echo "================================================="
echo "🎯 Testing 3 LR scales × ${#SHOTS_LIST[@]} shot sizes = $((3 * ${#SHOTS_LIST[@]})) total experiments"
echo "📊 LR scales: HIGH (~1e-5), MEDIUM (~1e-6), LOW (~1e-7)"
echo "⏰ Epochs: HIGH=fewer, MEDIUM=moderate, LOW=more"
echo "🌱 Seed: $TEST_SEED (fixed for all tests)"
echo ""

# Function to modify train3.py for specific hyperparameter configuration
modify_hyperparameters() {
    local config_name=$1
    local shots=$2
    
    echo "🔧 Configuring $config_name hyperparameters for $shots-shot..."
    
    # Create the hyperparameter configuration based on config_name
    case $config_name in
        "HIGH")
            case $shots in
                2|4)
                    lr_factor="1.0"
                    epoch_factor="0.8"
                    patience="5"
                    dropout="0.05"
                    warmup="0.05"
                    ;;
                8|16)
                    lr_factor="0.8"
                    epoch_factor="0.9"
                    patience="8"
                    dropout="0.05"
                    warmup="0.1"
                    ;;
                32|64)
                    lr_factor="0.6"
                    epoch_factor="1.0"
                    patience="10"
                    dropout="0.1"
                    warmup="0.1"
                    ;;
                128)
                    lr_factor="0.5"
                    epoch_factor="1.0"
                    patience="12"
                    dropout="0.1"
                    warmup="0.1"
                    ;;
            esac
            ;;
        "MEDIUM")
            case $shots in
                2|4)
                    lr_factor="0.1"
                    epoch_factor="1.2"
                    patience="8"
                    dropout="0.1"
                    warmup="0.15"
                    ;;
                8|16)
                    lr_factor="0.08"
                    epoch_factor="1.3"
                    patience="10"
                    dropout="0.1"
                    warmup="0.2"
                    ;;
                32|64)
                    lr_factor="0.06"
                    epoch_factor="1.4"
                    patience="12"
                    dropout="0.15"
                    warmup="0.2"
                    ;;
                128)
                    lr_factor="0.05"
                    epoch_factor="1.5"
                    patience="15"
                    dropout="0.15"
                    warmup="0.25"
                    ;;
            esac
            ;;
        "LOW")
            case $shots in
                2|4)
                    lr_factor="0.01"
                    epoch_factor="1.5"
                    patience="12"
                    dropout="0.15"
                    warmup="0.25"
                    ;;
                8|16)
                    lr_factor="0.008"
                    epoch_factor="1.6"
                    patience="15"
                    dropout="0.2"
                    warmup="0.3"
                    ;;
                32|64)
                    lr_factor="0.006"
                    epoch_factor="1.8"
                    patience="18"
                    dropout="0.2"
                    warmup="0.3"
                    ;;
                128)
                    lr_factor="0.005"
                    epoch_factor="2.0"
                    patience="20"
                    dropout="0.25"
                    warmup="0.35"
                    ;;
            esac
            ;;
    esac
    
    # Calculate effective values
    effective_lr=$(python3 -c "print(f'{$BASE_LR * $lr_factor:.2e}')")
    effective_epochs=$(python3 -c "print(int($BASE_EPOCHS * $epoch_factor))")
    
    echo "   📊 LR: $BASE_LR × $lr_factor = $effective_lr"
    echo "   📊 Epochs: $BASE_EPOCHS × $epoch_factor = $effective_epochs"
    echo "   📊 Patience: $patience | Dropout: $dropout | Warmup: $warmup"
    
    # Modify the train3.py file with these hyperparameters
    python3 -c "
import re

# Read the file
with open('$SCRIPT_PATH', 'r') as f:
    content = f.read()

# Find the calculate_adaptive_hyperparameters function and replace the $shots case
pattern = r'if train_size <= 4:.*?warmup_ratio = [0-9.]+.*?elif train_size <= 8:'
if $shots <= 4:
    replacement = '''if train_size <= 4:  # $shots-shot - TEST: $config_name LR
        lr_factor = $lr_factor  # TEST $config_name: $BASE_LR → $effective_lr
        epoch_factor = $epoch_factor  # $config_name epochs: $BASE_EPOCHS → $effective_epochs
        patience = $patience
        min_evals = 3
        dropout_rate = $dropout
        warmup_ratio = $warmup
    elif train_size <= 8:'''
elif $shots <= 8:
    pattern = r'elif train_size <= 8:.*?warmup_ratio = [0-9.]+.*?elif train_size <= 16:'
    replacement = '''elif train_size <= 8:  # $shots-shot - TEST: $config_name LR
        lr_factor = $lr_factor  # TEST $config_name: $BASE_LR → $effective_lr
        epoch_factor = $epoch_factor  # $config_name epochs: $BASE_EPOCHS → $effective_epochs
        patience = $patience
        min_evals = 4
        dropout_rate = $dropout
        warmup_ratio = $warmup
    elif train_size <= 16:'''
elif $shots <= 16:
    pattern = r'elif train_size <= 16:.*?warmup_ratio = [0-9.]+.*?elif train_size <= 32:'
    replacement = '''elif train_size <= 16:  # $shots-shot - TEST: $config_name LR
        lr_factor = $lr_factor  # TEST $config_name: $BASE_LR → $effective_lr
        epoch_factor = $epoch_factor  # $config_name epochs: $BASE_EPOCHS → $effective_epochs
        patience = $patience
        min_evals = 5
        dropout_rate = $dropout
        warmup_ratio = $warmup
    elif train_size <= 32:'''
elif $shots <= 32:
    pattern = r'elif train_size <= 32:.*?warmup_ratio = [0-9.]+.*?elif train_size <= 64:'
    replacement = '''elif train_size <= 32:  # $shots-shot - TEST: $config_name LR
        lr_factor = $lr_factor  # TEST $config_name: $BASE_LR → $effective_lr
        epoch_factor = $epoch_factor  # $config_name epochs: $BASE_EPOCHS → $effective_epochs
        patience = $patience
        min_evals = 6
        dropout_rate = $dropout
        warmup_ratio = $warmup
    elif train_size <= 64:'''
elif $shots <= 64:
    pattern = r'elif train_size <= 64:.*?warmup_ratio = [0-9.]+.*?else:'
    replacement = '''elif train_size <= 64:  # $shots-shot - TEST: $config_name LR
        lr_factor = $lr_factor  # TEST $config_name: $BASE_LR → $effective_lr
        epoch_factor = $epoch_factor  # $config_name epochs: $BASE_EPOCHS → $effective_epochs
        patience = $patience
        min_evals = 7
        dropout_rate = $dropout
        warmup_ratio = $warmup
    else:'''
else:  # 128+
    pattern = r'else:  # 128\+ shot.*?warmup_ratio = [0-9.]+\s+'
    replacement = '''else:  # $shots+ shot - TEST: $config_name LR
        lr_factor = $lr_factor  # TEST $config_name: $BASE_LR → $effective_lr
        epoch_factor = $epoch_factor  # $config_name epochs: $BASE_EPOCHS → $effective_epochs
        patience = $patience
        min_evals = 8
        dropout_rate = $dropout
        warmup_ratio = $warmup
    '''

# Apply the replacement
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open('$SCRIPT_PATH', 'w') as f:
    f.write(new_content)

print('✅ Hyperparameters updated for $config_name configuration')
"
}

# Main testing loop
for SHOTS in "${SHOTS_LIST[@]}"; do
    echo ""
    echo "🎯 TESTING $SHOTS-SHOT ACROSS 3 LR SCALES"
    echo "----------------------------------------"
    
    TRAIN_FILE="$TRAIN_DIR/train_${SHOTS}_seed${TEST_SEED}.json"
    
    if [ ! -f "$TRAIN_FILE" ]; then
        echo "⚠️  Skipping: $TRAIN_FILE not found"
        continue
    fi
    
    # Test each configuration
    for CONFIG in "HIGH" "MEDIUM" "LOW"; do
        echo ""
        echo "🧪 Testing $SHOTS-shot with $CONFIG LR configuration..."
        
        # Modify hyperparameters for this configuration
        modify_hyperparameters "$CONFIG" "$SHOTS"
        
        # Set output directory
        OUTPUT_DIR="/data/Amin/Models/${MODEL_NAME_CLEAN}/test_${SHOTS}shot_${CONFIG}_lr"
        mkdir -p "$OUTPUT_DIR"
        
        # Adaptive batch sizes
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
        
        echo "🚀 Starting training: $SHOTS-shot $CONFIG LR (seed $TEST_SEED)..."
        echo "📄 File: $TRAIN_FILE"
        echo "⚙️  Batch: $BATCH_SIZE (micro: $MICRO_BATCH)"
        echo "📦 Output: $OUTPUT_DIR"
        
        # Run training
        python "$SCRIPT_PATH" \
            --base_model "$BASE_MODEL" \
            --COT_train_file "$TRAIN_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --training_losses_file "$LOSSES_DIR" \
            --batch_size $BATCH_SIZE \
            --micro_batch_size $MICRO_BATCH \
            --num_epochs $BASE_EPOCHS \
            --learning_rate $BASE_LR \
            --weight_decay 0.01 \
            --max_grad_norm 1.0 \
            --label_smoothing_factor 0.1 \
            --warmup_steps 8 \
            --random_seed $TEST_SEED
        
        echo "✅ Completed: $SHOTS-shot $CONFIG LR"
        echo "📊 Check training_reports.txt for results"
        
        # Brief pause between configurations
        sleep 2
    done
    
    echo ""
    echo "🎯 Finalizing $SHOTS-shot analysis across all LR scales..."
    python3 -c "
import sys
import os
sys.path.insert(0, '/home/hq6375/Desktop/Code/Multi-Agent-Project')
os.chdir('/home/hq6375/Desktop/Code/Multi-Agent-Project')

try:
    from train3 import ShotReportManager
    manager = ShotReportManager()
    manager.finalize_shot_reports(${SHOTS})
    print('✅ $SHOTS-shot report finalization completed')
except Exception as e:
    print(f'❌ Error finalizing $SHOTS-shot reports: {e}')
"
    
    echo "✅ $SHOTS-shot testing complete (all 3 LR scales tested)"
done

echo ""
echo "🏁 SYSTEMATIC HYPERPARAMETER TESTING COMPLETE"
echo "=============================================="
echo "📊 Total experiments: $((3 * ${#SHOTS_LIST[@]}))"
echo "📈 Results: training_reports.txt"
echo "🐛 Issues: training_issues.txt"
echo ""
echo "📋 SUMMARY OF TESTED CONFIGURATIONS:"
echo "   HIGH LR: Aggressive learning, fewer epochs, minimal regularization"
echo "   MEDIUM LR: Balanced learning, moderate epochs, standard regularization"  
echo "   LOW LR: Conservative learning, more epochs, heavy regularization"
echo ""
echo "🎯 Next steps: Analyze training_reports.txt to identify optimal LR scale per shot size" 