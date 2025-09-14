
# Shell script to install and run all the needed things 

# Save the start time
start_time=$(date +%s)

#export CUDA_LAUNCH_BLOCKING=1

python -m pip install --upgrade pip

pip install -r requirements.txt

pip install --upgrade torch torchvision torchaudio

python -c "import torch, numpy, pandas, yfinance, talib; print('All dependencies installed successfully!')"

python testGPU.py

python crypto_rl_trader.py train default 500 test_model.pth

#python main.py



# Calculate the end time
end_time=$(date +%s)
# Calculate the difference
elapsed_time=$(( end_time - start_time ))
# Print the elapsed time
echo "Script execution time: $elapsed_time seconds"
