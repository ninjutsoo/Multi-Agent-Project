#!/bin/bash

# === COMPREHENSIVE HYPERPARAMETER GRID TEST ===
# 7 shot sizes × 3 LR scales = 21 experiments
# Seed 0 only, LR scales: e5, e6, e7

TRAIN_DIR="split_json_unbalanced"
SCRIPT_PATH="train3.py"
BASE_MODEL="emilyalsentzer/Bio_ClinicalBERT"
MODEL_DIR="/data/Amin/Models/$(echo "$BASE_MODEL" | tr '/:' '_')"

# Test configuration
SEED=0
SHOTS=(2 4 8 16 32 64 128)
LR_SCALES=("e5" "e6" "e7")

# LR and epoch configuration
declare -A LR_VALUES=( ["e5"]="1e-5" ["e6"]="1e-6" ["e7"]="1e-7" )
declare -A EPOCHS=( ["e5"]="40" ["e6"]="60" ["e7"]="80" )

# Fixed hyperparameters
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
LABEL_SMOOTHING=0.1

# Report file
REPORT_FILE="grid_test_comprehensive_report.txt"

echo "🧪 HYPERPARAMETER GRID TEST"
echo "=========================="
echo "Shots: ${SHOTS[*]}"
echo "LR scales: e5(${LR_VALUES[e5]},${EPOCHS[e5]}ep) e6(${LR_VALUES[e6]},${EPOCHS[e6]}ep) e7(${LR_VALUES[e7]},${EPOCHS[e7]}ep)"
echo "Seed: $SEED"
echo "Total experiments: $((${#SHOTS[@]} * ${#LR_SCALES[@]}))"
echo "Report: $REPORT_FILE"
echo ""

# Initialize comprehensive report
{
    echo "COMPREHENSIVE GRID TEST REPORT"
    echo "=============================="
    echo "Generated: $(date)"
    echo "Model: $BASE_MODEL"
    echo "Seed: $SEED"
    echo "LR Scales: e5(${LR_VALUES[e5]},${EPOCHS[e5]}ep) e6(${LR_VALUES[e6]},${EPOCHS[e6]}ep) e7(${LR_VALUES[e7]},${EPOCHS[e7]}ep)"
    echo ""
} > "$REPORT_FILE"

# Clear logs
> training_progress.txt
> experiment_logs.txt

run_count=0
success_count=0

for shots in "${SHOTS[@]}"; do
    echo "🎯 ${shots}-shot experiments"
    
    # Batch size configuration
    case $shots in
        2) batch=2; micro=1 ;;
        4) batch=4; micro=2 ;;
        8|16) batch=8; micro=2 ;;
        *) batch=16; micro=4 ;;
    esac
    
    for lr_scale in "${LR_SCALES[@]}"; do
        run_count=$((run_count + 1))
        
        lr=${LR_VALUES[$lr_scale]}
        epochs=${EPOCHS[$lr_scale]}
        
        train_file="$TRAIN_DIR/train_${shots}_seed${SEED}.json"
        output_dir="$MODEL_DIR/grid_${shots}shot_${lr_scale}_seed${SEED}"
        
        echo "🚀 RUN $run_count: ${shots}-shot ${lr_scale} (LR=$lr, Epochs=$epochs)"
        
        if [ ! -f "$train_file" ]; then
            echo "❌ SKIP: $train_file not found"
            continue
        fi
        
        mkdir -p "$output_dir"
        
        python "$SCRIPT_PATH" \
            --base_model "$BASE_MODEL" \
            --COT_train_file "$train_file" \
            --COT_dev_file "$TRAIN_DIR/validation.json" \
            --output_dir "$output_dir" \
            --batch_size $batch \
            --micro_batch_size $micro \
            --num_epochs $epochs \
            --learning_rate $lr \
            --weight_decay $WEIGHT_DECAY \
            --max_grad_norm $MAX_GRAD_NORM \
            --label_smoothing_factor $LABEL_SMOOTHING \
            --random_seed $SEED \
            --experiment_name "${shots}shot_${lr_scale}" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            success_count=$((success_count + 1))
            echo "✅ SUCCESS"
        else
            echo "❌ FAILED"
        fi
        echo ""
    done
    
    # Generate comprehensive report section for this shot
    echo "📊 Adding ${shots}-shot analysis to report..."
    {
        echo "============================================"
        echo "${shots}-SHOT EXPERIMENTS ANALYSIS"
        echo "============================================"
        echo ""
        
        for lr_scale in "${LR_SCALES[@]}"; do
            echo "--- ${lr_scale} LR (${LR_VALUES[$lr_scale]}, ${EPOCHS[$lr_scale]} epochs) ---"
            
            # Extract training loss series
            train_losses=($(grep "^${shots}shot_${lr_scale},TRAIN," training_progress.txt | cut -d',' -f4 | cut -d'=' -f2))
            eval_losses=($(grep "^${shots}shot_${lr_scale},EVAL," training_progress.txt | cut -d',' -f4 | cut -d'=' -f2))
            aurocs=($(grep "^${shots}shot_${lr_scale},EVAL," training_progress.txt | cut -d',' -f5 | cut -d'=' -f2))
            best_auroc=$(grep "^${shots}shot_${lr_scale},FINAL," training_progress.txt | cut -d',' -f3 | cut -d'=' -f2)
            
            echo "Training Loss Series: [$(IFS=','; echo "${train_losses[*]}")]"
            echo "Eval Loss Series:     [$(IFS=','; echo "${eval_losses[*]}")]"
            echo "AUROC Series:         [$(IFS=','; echo "${aurocs[*]}")]"
            echo "Best AUROC:           ${best_auroc:-N/A}"
            echo "Total Evaluations:    ${#eval_losses[@]}"
            
            # Loss trend analysis
            if [ ${#train_losses[@]} -gt 1 ]; then
                first_loss=${train_losses[0]}
                last_loss=${train_losses[-1]}
                echo "Loss Trend:           ${first_loss} → ${last_loss} ($(python3 -c "print('IMPROVED' if float('$last_loss') < float('$first_loss') else 'WORSENED')" 2>/dev/null || echo 'UNKNOWN'))"
            fi
            
            echo ""
        done
        
        echo "COMPARISON SUMMARY:"
        echo "Best AUROC by LR:"
        for lr_scale in "${LR_SCALES[@]}"; do
            best_auroc=$(grep "^${shots}shot_${lr_scale},FINAL," training_progress.txt | cut -d',' -f3 | cut -d'=' -f2)
            echo "  ${lr_scale}: ${best_auroc:-N/A}"
        done
        echo ""
        
    } >> "$REPORT_FILE"
    
    echo "✅ ${shots}-shot analysis added to report"
    echo ""
done

# Final summary
{
    echo "============================================"
    echo "FINAL GRID TEST SUMMARY"
    echo "============================================"
    echo "Total runs: $run_count"
    echo "Successful: $success_count"
    echo "Failed: $((run_count - success_count))"
    echo ""
    echo "All models saved to: $MODEL_DIR/grid_*"
    echo "Training logs: training_progress.txt"
    echo "Experiment configs: experiment_logs.txt"
    echo ""
    echo "Test completed: $(date)"
} >> "$REPORT_FILE"

echo "🏁 GRID TEST COMPLETE"
echo "===================="
echo "Total runs: $run_count"
echo "Successful: $success_count"
echo "Failed: $((run_count - success_count))"
echo ""
echo "📁 Files generated:"
echo "   $REPORT_FILE - Comprehensive analysis report"
echo "   training_progress.txt - Raw training logs"
echo "   experiment_logs.txt - Experiment configurations"
echo ""
echo "📦 Models saved to: $MODEL_DIR/grid_*" 