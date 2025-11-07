# Comp4471_Project

Branch for fine tune test on TrOCR with HASYv2

with 8GB VRAM CUDA GPU, training time for 1 small fold (A-Z, a-z, pi, alpha, beta, sigma, sum) is 5 hours (see train_runtime in train-0.log)

2 fine-tuned model, one with argument tweaks from default Seq2SeqTrainingArguments (./trocr-hasyv20), one without (./trocr-hasyv2-0)
_trocr-hasyv2-00 could obtain 27% accuracy
_trocr-hasyv2-0 could only obtain 9% accuracy

Note that safetensors and optimizer.pt are not committed as LFS is required
Full models are temporarily saved at hf https://hf.co/richyzj/trocr_hasyv2