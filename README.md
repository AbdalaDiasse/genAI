# genAI
llm-lvm-playground/
├── training/              # Fine-tuning & pretraining workflows for LLMs and LVMs
│   ├── README.md          # Overview of training examples, setup, and tips 
│   ├── llm/               # Training scripts/notebooks for language models 
│   └── lvm/               # Training scripts/notebooks for vision models 
├── inference/             # Inference pipelines and serving 
│   ├── README.md          # Inference guide using Transformers, vLLM, DeepSpeed, etc.
│   └── ...                # Example scripts for different frameworks 
├── quantization/          # Model quantization tools and examples
│   ├── README.md          # Using bitsandbytes, GPTQ, AWQ, etc. for compression
│   └── ...                # Scripts for quantizing models 
├── deployment/            # Deployment examples (cloud, on-prem, edge)
│   ├── README.md          # Guide to containerization, FastAPI, Triton, ONNX/TensorRT
│   └── ...                # Configs/scripts for various deployment scenarios 
├── evaluation/            # Evaluation and benchmarking tools
│   ├── README.md          # Prompts, metrics, and performance benchmarking 
│   └── ...                # Example evaluation scripts and notebooks 
└── model_zoo/             # Index of recommended open-source models 
    ├── README.md          # Catalog of LLMs and LVMs with usage guides 
    └── ...                # (Potential subfolders or files per model category)
