# RAIHFT: Recruitment AI Human-feedback Toolkit

## Purpose

RAIHFT is an open-source toolkit that enables AI teams at recruiting companies to align and evaluate language and decision models using Reinforcement Learning from Human Feedback (RLHF). It provides a complete pipeline for collecting recruiter preferences (data labelling), training a reward model, fine-tuning models with RL, and analysing model quality and fairness. By combining cutting-edge RLHF methods with bias detection and a user-centred interface, RAIHFT helps ensure recruitment AI systems behave as intended and ethically.

## Key Features

- **RLHF Pipeline**: Train a reward model from human preference data and fine-tune an AI policy (e.g. recruiter assistant) using algorithms like PPO or newer methods such as DPO.
- **Data Labelling & Annotation**: Tools for recruiters to label candidate profiles or model responses (e.g. thumbs-up/down) to generate preference data for training.
- **Bias Detection & Mitigation**: Built-in metrics (e.g. disparate impact, equal opportunity) and debiasing methods (pre-processing reweighting, adversarial debiasing) to measure and reduce unfairness.
- **Evaluation Analytics**: Functions to log performance (reward scores, accuracy) and fairness metrics, enabling tracking of model improvements.
- **Confidence-building UX Tool**: An AI-powered CLI or web interface that provides skill diagnostics, positive affirmations, and explanatory feedback, helping users (e.g. recruiters or junior ML engineers) understand model decisions and build confidence.

## Setup (Google Colab)

1. Clone the repo into a Colab or local environment.
2. Install dependencies: Python ≥3.8, PyTorch, Hugging Face `transformers`, `trl` (for PPO/DPO), `fairlearn`/`aif360`, and other libs (`pip install -r requirements.txt`).
3. Launch provided Jupyter/Colab notebooks. For example, run `colab_notebooks/rlhf_training.ipynb` which sets up a small-scale RLHF example using a public LLM and dummy preference data.
4. The toolkit is designed to run on free GPU instances (e.g. Colab GPU or Kaggle Kernels), with small model examples for demos.

## Usage Example

- Run `python data_labeling_app.py` to start a web labelling interface for recruiters to compare AI outputs.
- Execute `python train_reward_model.py` to train the reward model on collected labels.
- Use `python train_rl_agent.py` to fine-tune the base model via PPO or DPO, logging metrics.
- Access the bias report by running `python bias_analysis.py` on validation data; it prints metrics like statistical parity difference.
- Launch `python confidence_tool.py` to start the confidence-building CLI (it will ask about your skills and provide feedback).

## Contributing

We welcome contributions! Please fork the repository, create a feature branch, and submit a pull request. Follow the style guidelines in `CONTRIBUTING.md`. Write unit tests for new functionality—report issues via the GitHub Issue tracker. Engage with the community through discussions to help prioritise features.

## License

This project is licensed under the MIT License, promoting broad reuse by enterprise teams and researchers.
