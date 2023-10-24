# Accompanying Code for "Reducing Simulator Calls in SBI"

This repository contains the code accompanying the paper "Reducing Simulator Calls in SBI" by David Refaeli and David M. Steinberg. The provided Python and Jupyter Notebook files are organized to help you navigate through the implementation of the proposed methods.

## Getting Started

To run the toy tasks, refer to the `RunTasks.ipynb` notebook. This notebook provides a comprehensive overview of the implementation and execution of the toy tasks.

For the individual code of tasks 1 to 6, consult the following files:
- `tasks/GMM.py`
- `tasks/TwoMoons.py`
- `tasks/Sisson.py`
- `tasks/SLCP.py`
- `tasks/BayesLR.py`
- `tasks/BerGLM.py`

Each file corresponds to a specific task and contains the relevant code for execution.

To run the Sequential Neural Likelihood Estimation (SNLE) vs. Surrogate method, refer to the `SNLE.ipynb` notebook. This notebook guides you through the steps to execute the SNLE implementation.

For the Hodgkin-Huxley (HH), follow the instructions in the `HH.ipynb` notebook. This notebook provides details on running the HH problem.

## Dependencies

Make sure you have the necessary dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Citation

If you use this code in your research, please consider citing the original paper:

```
@article{reducing-simulator-calls-sbi,
  author    = {David Refaeli and David M. Steinberg},
  title     = {Reducing Simulator Calls in SBI},
  journal   = {Journal of Scientific Computing},
  year      = {YYYY},
  volume    = {XX},
  number    = {XX},
  pages     = {XXX-XXX},
  doi       = {XXX.XXXX/XXXXXXX}
}
```

Feel free to explore and experiment with the code for a better understanding of the techniques discussed in the paper. If you encounter any issues or have questions, please don't hesitate to open an issue in this repository.

Happy exploring!
