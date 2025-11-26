#!/bin/bash
#
# Quick test to verify GSPO fix
#

echo "Testing GSPO fix with a minimal training run..."

python run_semantic_segmentation.py \
    --task isic2017 \
    --seed 42686693 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --gspo_group_size 4 \
    --output_dir outputs/gspo_fix_test \
    --batch_size 2 \
    --per_gpu_batch_size 1 &

# Get the PID
PID=$!

echo "Training started with PID: $PID"
echo "Waiting for 60 seconds to check if training runs without errors..."

# Wait 60 seconds
sleep 60

# Check if process is still running
if ps -p $PID > /dev/null; then
    echo "✓ SUCCESS: Training is running without errors!"
    echo "Stopping the test training..."
    kill $PID
    wait $PID 2>/dev/null
    echo "Test completed successfully!"
    exit 0
else
    echo "✗ FAILED: Training crashed. Check the logs for errors."
    exit 1
fi

