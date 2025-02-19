subjects=(
    "Algebra"
    "Abstract_algebra"
    "Combinatorics"
    "Statistics"
    "Geometry"
    "Set_theory_and_logic"
    "Probability"
    "Arithmetic"
    "Trigonometry"
    "Complex_analysis"
    "Differential_equations"
    "Calculus_-_single_variable"
    "Linear_algebra"
    "Calculus_-_multivariable"
    "Number_theory"
    "Financial_mathematics"
)

models=(
    "Meta-Llama-3-8B-Instruct"
)

for model in "${models[@]}"; do
    for subject in "${subjects[@]}"; do
        python generate_open.py --model_path /home/math/models/$model \
                                --subject $subject \
                                --prompt llama3math \
                                --tensor_parallel_size 4
        python eval_rule.py --model_path /home/math/models/$model --subject $subject
    done
    python eval_rule.py --model_path /home/math/models/$model
done