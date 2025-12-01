# Comp4471_Project

## Parent Branch(richy-small-dataset-fitting) for fine tune test on TrOCR with HASYv2

with 8GB VRAM CUDA GPU, training time for 1 small fold (A-Z, a-z, pi, alpha, beta, sigma, sum) is 5 hours (see train_runtime in train-0.log)

2 fine-tuned model, one with argument tweaks from default Seq2SeqTrainingArguments (./trocr-hasyv20), one without (./trocr-hasyv2-0)
_trocr-hasyv2-00 could obtain 27% accuracy
_trocr-hasyv2-0 could only obtain 9% accuracy

Note that safetensors and optimizer.pt are not committed as LFS is required
Full models are temporarily saved at hf https://hf.co/richyzj/trocr_hasyv2

## Child Branch(HME-v2) for finetuning on TrOCR with HME100kk

with Colab A100 High-RAM(167GB RAM, 80GB VRAM), training time for 12000 training data for 10 epoch is 7 hours, 20000 for 15 epoch is 19 hours

Result of 12000, 10 epoch is in the notebook, result training loss of 0.93; 20000 for 15 epoch is still under progress.

Full models are saved at kaggle https://www.kaggle.com/models/richyz/trocr-small-handwritten-pretrained

