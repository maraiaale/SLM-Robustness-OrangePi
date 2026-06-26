# Model archives

Model binaries are not stored directly in this GitHub repository because of storage and file-size limits.

The model files are provided separately through external archives:

- CPU GGUF models: https://drive.google.com/file/d/1dCq04N_Tcii0eFAmfcKSMUhGk-F5yyDs/view?usp=sharing
- NPU RKLLM models: https://drive.google.com/file/d/1dAIzqZCaDYPECI0bCHx9ClTbCbAFl_Hg/view?usp=sharing

After downloading the archives, extract them into the following directories:

- CPU GGUF models into `models/cpu_gguf/`
- NPU RKLLM models into `models/npu_rkllm/`

Expected CPU GGUF files:

- `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- `qwen2.5-1.5b.gguf`
- `qwen2.5-3b.gguf`
- `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
- `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`

Expected NPU RKLLM files:

- `qwen2.5-0.5b-instruct-rk3588-w8a8.rkllm`
- `qwen2.5-1.5b-instruct-rk3588-w8a8.rkllm`
- `qwen2.5-3b-instruct-rk3588.rkllm`
- `qwen2.5-7b-instruct-rk3588.rkllm`
