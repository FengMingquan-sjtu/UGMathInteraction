py_script="generate_dual_close.py"
subject="Differential_equations"
output_dir="./results/human_assist"
nproc=16
n_sample=300 


# gpt-4.1
url_0="https://api.toiotech.com/v1/chat/completions"
key_0="sk-"  # your key here
model_0="gpt-4.1"
prompt_0="deepseekmath-assist"

# o4-mini
url_1="https://api.toiotech.com/v1/chat/completions"
key_1="sk-"  # your key here
model_1="o4-mini"
prompt_1="deepseekmath-human"

for ((i=1; i<=5; i++)); do
    echo "Iteration $i, phase 1"
    export OPENAI_BASE_URL=$url_1
    export OPENAI_API_KEY=$key_1
    python $py_script --model $model_1 --subject $subject --prompt $prompt_1 --nproc $nproc --phase 1 --output_dir $output_dir --test_part 0 $n_sample

    echo "Iteration $i, phase 0"
    export OPENAI_BASE_URL=$url_0
    export OPENAI_API_KEY=$key_0
    python $py_script --model $model_0 --subject $subject --prompt $prompt_0 --nproc $nproc --phase 0 --output_dir $output_dir --test_part 0 $n_sample
done

