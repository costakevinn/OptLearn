from examples.simple_examples import (
    run_quadratic_example,
    run_rosenbrock_example,
    run_himmelblau_example
)
from examples.nn_example import run_nn_example  # <-- seu novo exemplo

def main():
    examples = [
        ("Quadratic", run_quadratic_example),
        ("Rosenbrock", run_rosenbrock_example),
        ("Himmelblau", run_himmelblau_example),
        ("Neural Network", run_nn_example)  # <-- adiciona a NN aqui
    ]

    print("\n==== Starting Optimization Examples ====")
    for name, func in examples:
        print(f"\n==== Starting {name} Example ====")
        func()
        print(f"\n==== Completed {name} Example ====")
    print("\n==== Completed All Examples ====")

if __name__ == "__main__":
    main()
