# LLM & LVM Playground Monorepo

Welcome to the **LLM & LVM Playground**, a comprehensive monorepo template for experimenting with **Large Language Models (LLMs)** and **Large Vision Models (LVMs)**.

This repo is designed for **intermediate to advanced machine learning practitioners** who want to explore training, inference, quantization, deployment, and benchmarking across cloud, local, and edge environments.

---

## 📁 Repository Structure

```plaintext
llm-lvm-playground/
├── training/
│   ├── llm/                  # Fine-tuning and pretraining of LLMs
│   └── lvm/                  # Fine-tuning of LVMs (ViT, Diffusers, etc.)
│
├── inference/
│   ├── llm/                  # Inference with Hugging Face, vLLM, DeepSpeed
│   └── lvm/                  # Inference for vision models
│
├── quantization/
│   ├── bitsandbytes/         # 8-bit / 4-bit quantization examples
│   ├── gptq/                 # GPTQ quantized models
│   └── awq/                  # Activation-aware weight quantization
│
├── deployment/
│   ├── cloud/                # AWS, GCP deployment (EC2, SageMaker, Vertex AI)
│   ├── local/                # FastAPI serving with Docker
│   ├── edge/                 # ONNX / TensorRT for Jetson, Raspberry Pi
│   ├── triton/               # NVIDIA Triton Inference Server
│   └── onnx_tensorrt/        # ONNX → TensorRT optimized inference
│
├── evaluation/
│   ├── prompts/              # Prompt sets for evaluation
│   ├── benchmark.py          # Performance and latency tests
│   ├── eval_harness.py       # EleutherAI Evaluation Harness
│   └── results/              # Output logs and metrics
│
├── model_zoo/
│   ├── llms.md               # Recommended LLMs with quick usage guide
│   └── lvms.md               # Recommended LVMs with quick usage guide
│
└── README.md                 # You are here
```

---

## 🚀 Key Features

- 🔧 **Training**: Fine-tuning and pretraining for LLaMA, Mistral, Falcon, ViT, SD
- 🧠 **Inference**: Serve models with Hugging Face, vLLM, DeepSpeed, TGI
- 📦 **Quantization**: Use bitsandbytes (4/8-bit), GPTQ, AWQ to optimize models
- 📡 **Deployment**: Cloud, local server (FastAPI), and edge device setups
- 📊 **Evaluation**: Evaluate with EleutherAI Harness and benchmark scripts
- 🧭 **Model Zoo**: Curated list of top-performing open-source LLMs & LVMs

---

## 🛠️ Setup Instructions

```bash
# Clone this repository
git clone https://github.com/your-org/llm-lvm-playground.git
cd llm-lvm-playground

# (Optional) Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies (submodules have their own requirements.txt files)
pip install -r requirements.txt
```

Each module (`training/`, `inference/`, etc.) has its own README with usage instructions.

---

## 📚 Documentation Index

- [`training/`](./training/) – Fine-tune or pretrain LLMs and LVMs  
- [`inference/`](./inference/) – Efficient inference with vLLM, Hugging Face, DeepSpeed  
- [`quantization/`](./quantization/) – Quantize with bitsandbytes, GPTQ, AWQ  
- [`deployment/`](./deployment/) – Deploy on cloud, locally or on edge devices  
- [`evaluation/`](./evaluation/) – Run eval harness, measure latency/memory  
- [`model_zoo/`](./model_zoo/) – Curated list of top open-source models with examples  

---

## 🤝 Contributing

We welcome all contributions!

- Add more fine-tuning or inference examples
- Improve deployment guides or Docker setups
- Contribute benchmark results or evaluations
- Submit issues or feature requests

Please submit pull requests against `main` with a clear description of the change.

---

## 📝 License

This project is licensed under the **Apache 2.0 License**.

Note: Refer to each model’s specific license in the `model_zoo/` directory when using or modifying individual models.

---

## 🙌 Credits

This repo builds on the amazing work from:

- [Hugging Face](https://huggingface.co/)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness)
- [Meta AI](https://ai.meta.com/)
- [Stability AI](https://stability.ai/)
- [NVIDIA](https://developer.nvidia.com/triton-inference-server)

---

**Maintainer**: [Your Name / Org]  
📫 Contact: [your-email@example.com]  
⭐ Star the repo if you find it useful!


Let me know if you want this packaged as a `.zip` with placeholder folders and README files included.
