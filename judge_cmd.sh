output_dir=./results/o4-mini  #modify this to your output directory
export OPENAI_BASE_URL=https://api.toiotech.com/v1
export OPENAI_API_KEY=sk-  # your key here
python eval_marj.py --model gpt-4.1 --subject Differential_equations --output_dir $output_dir