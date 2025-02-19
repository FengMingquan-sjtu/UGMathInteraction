subjects=(
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
    "Algebra"
    "Number_theory"
    "Financial_mathematics"
)

models=(
    gpt-4o-2024-08-06
)

for model in "${models[@]}"; do
    for subject in "${subjects[@]}"; do
        python generate_close.py --model $model \
                                --subject $subject \
                                --prompt raw \
                                --nproc 16

        python eval_marj.py --model_path $model --subject $subject
    done
    python eval_marj.py --model_path $model
done