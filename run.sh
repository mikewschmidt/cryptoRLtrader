
# Shell script to install and run all the needed things 

# Save the start time
start_time=$(date +%s)

#export CUDA_LAUNCH_BLOCKING=1

python -m pip install --upgrade pip

pip install -r requirements.txt

pip install --upgrade torch torchvision torchaudio

python -c "import torch, numpy, pandas, yfinance, talib; print('All dependencies installed successfully!')"

python testGPU.py

'''
Episodes (Training Duration)

500-1000: Quick testing, basic models
1000-2000: Standard production use
3000-5000: High-performance models
5000+: Research/experimental (diminishing returns)
'''

#python crypto_rl_trader.py train default 500 test_model.pth

#python crypto_rl_trader.py train 1_months_of_days_of_crypto_1m.csv 2000 test_model.pth
python crypto_rl_trader.py train 3_months_of_days_of_crypto_1m.csv 5000 RL_model_5000_3month.pth
python crypto_rl_trader.py train 3_months_of_days_of_crypto_1m.csv 1000 RL_model_1000_3month.pth


# Calculate the end time
end_time=$(date +%s)
# Calculate the difference
elapsed_time=$(( end_time - start_time ))
# Print the elapsed time
echo "Script execution time: $elapsed_time seconds"
