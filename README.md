## Getting Started

1. Install required packages, use `conda env create`
2. Activate the env via `conda activate multi_empowerment`

## Training

<!-- ```python train_grid.py --num_boxes 4 --seed 0``` -->

With Random Empowerment Policy and Noisy Greedy Human Policy and Random Human Policy on Multiagent Gridworld:
```python train_grid.py --num_boxes 4 --seed 0 --num_humans 2 --num_goals 2 --random```

## Troubleshooting
If running on Mac M series chip and using JAX metal, then type `export ENABLE_PJRT_COMPATIBILITY=1` in the terminal before running any training code.

