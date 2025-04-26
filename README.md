# Context-aware Evaluation of Machine Translation Systems (CAEMT)

Welcome to the repository for **Context-aware Evaluation of Machine Translation Systems (CAEMT)**.  
This project focuses on developing methods and resources for evaluating Machine Translation (MT) systems by incorporating **contextual information**, particularly visual features, into the evaluation process.

CAEMT utilizes **Jina** and **CLIP Vision-Language Models** to calculate **cross-language, cross-model similarities** between translation outputs and reference inputs.  
The goal is to create a system that:
- Works **independently** alongside existing text-based metrics
- Provides **visually-aware assessments** of MT quality
- Enhances evaluation by capturing multimodal cues often missed in traditional text-only evaluations

The following diagram illustrates the high-level workflow of CAEMT:

![CAEMT Overview](overview-CAEMT.png.png)
---

## Project Structure

This repository contains:
- Core scripts for similarity computation
- Model integration code (Jina and CLIP)
- Evaluation pipelines
- Configuration examples and sample data

## Getting Started

To set up CAEMT locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/caemt.git
   cd caemt
   ```

2. **Set up your environment**
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download/pre-load models**
   (Instructions to download Jina/CLIP models if required will be added here.)

4. **Run Example Evaluations**
   Example usage:
   ```bash
   python evaluate.py --config configs/sample_config.yaml
   ```

---

## Requirements

- Python 3.8+
- [Jina AI](https://github.com/jina-ai/jina)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- Torch
- NumPy, pandas
- (All dependencies listed in `requirements.txt`)

---

## Folder Structure

```
caemt/
│
├── data/               # Datasets and visual samples
├── metrics/            # Cross-modal evaluation metric code
├── models/             # Vision-language model wrappers
├── scripts/            # Evaluation and utility scripts
├── configs/            # YAML config files for evaluation setup
├── examples/           # Example usage scripts
├── README.md           # Project overview
└── requirements.txt    # Python dependencies
```

---

## Usage

You can use CAEMT in the following ways:
- As a **standalone metric** for multimodal translation evaluation
- **Integrated with** existing text-based metrics (like BLEU, COMET, BLEURT) for enhanced quality estimation
- For **analyzing cross-language visual semantic alignment**

Example evaluation command:
```bash
python evaluate.py --config configs/sample_config.yaml
```

---

## Contributing

Contributions are welcome! 🚀  
Feel free to open issues for feature requests or bug reports, and submit pull requests to improve the project.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Authors

- Sami Ul Haq, Sheila Castilho, Yvette Graham
