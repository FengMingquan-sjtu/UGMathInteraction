export OPENAI_BASE_URL=https://api.toiotech.com/v1/chat/completions
export OPENAI_API_KEY=sk- # your key here
model=o4-mini #replace with other model like gpt-4.1
python generate_close.py --model $model --subject Differential_equations --prompt deepseekmath --nproc 16