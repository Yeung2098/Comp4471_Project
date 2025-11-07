import os
from datasets import load_dataset
import numpy as np
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image
from pathlib import Path

def accuracy_metric(trainer, test_dataset):
    """Compute sequence-level exact-match accuracy.

    This decodes generated token ids and reference label ids to strings and
    computes the proportion of exact matches. Handles both generated sequences
    (predictions shape (N, seq_len)) and logits (N, seq_len, vocab) returned by
    the trainer.
    """
    preds_out = trainer.predict(test_dataset)
    pred_logits_or_ids = preds_out.predictions

    # If trainer returned logits (N, seq_len, vocab) take argmax over vocab dim
    if pred_logits_or_ids is None:
        print("No predictions were returned by trainer.predict().")
        return
    if pred_logits_or_ids.ndim == 3:
        pred_ids = np.argmax(pred_logits_or_ids, axis=2)
    elif pred_logits_or_ids.ndim == 2:
        # Already token ids (generated sequences)
        pred_ids = pred_logits_or_ids
    else:
        raise ValueError(f"Unexpected prediction shape: {pred_logits_or_ids.shape}")

    # Get reference labels from the dataset. They may still contain pad_token_id.
    labels = np.array(test_dataset["labels"])

    # Replace -100 (if present) with pad_token_id so decoding works; if pad_token_id
    # isn't set on the tokenizer this is a no-op.
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = -100
    labels_for_decode = labels.copy()
    labels_for_decode[labels_for_decode == -100] = pad_id

    # Sanitize token id arrays: replace any negative values (e.g. -100) with pad_id
    pred_ids = np.array(pred_ids, dtype=np.int64)
    labels_for_decode = np.array(labels_for_decode, dtype=np.int64)
    # Ensure we have a pad id to use for replacements
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        # fallback to eos or 0
        pad_id = getattr(processor.tokenizer, "eos_token_id", 0)
    pred_ids[pred_ids < 0] = pad_id
    labels_for_decode[labels_for_decode < 0] = pad_id

    # Convert rows to Python lists so the fast tokenizer doesn't see negative ints
    pred_seqs = [row.tolist() for row in pred_ids]
    label_seqs = [row.tolist() for row in labels_for_decode]

    # Decode token ids to strings
    decoded_preds = processor.tokenizer.batch_decode(pred_seqs, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(label_seqs, skip_special_tokens=True)

    # Sequence-level exact match
    matches = [p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)]
    accuracy = float(np.mean(matches)) if len(matches) > 0 else 0.0
    print(f"Accuracy (exact match): {accuracy:.4f}")
    return accuracy

def preprocess_function(examples):
    images = []
    for image_path in examples["path"]:
        p = Path(image_path)
        if not p.is_absolute():
            candidate = Path(dataset_name) / image_path
            if candidate.exists():
                p = candidate
            else:
                p = p.resolve()
        images.append(Image.open(str(p)).convert("RGB"))
    texts = examples["latex"]

    # Return numpy arrays / lists so the Dataset can serialize them. We'll convert to tensors in the collator.
    pixel_values = processor(images=images, return_tensors="np").pixel_values
    labels = processor.tokenizer(texts, padding="max_length", max_length=128, truncation=True).input_ids

    # Keep pad token ids for now; the collator will replace them with -100 for the loss.
    return {"pixel_values": pixel_values.tolist(), "labels": labels}

def custom_data_collator(batch):
        """Collate function to stack image pixel arrays and pad/mask label sequences.

        Expects each example to have keys:
            - 'pixel_values': nested list or numpy array (C,H,W)
            - 'labels': list[int]
        Returns a dict with torch tensors: 'pixel_values' and 'labels' (labels masked with -100).
        """
        # Stack pixel values (each entry is a numpy array or list)
        pixel_list = [np.array(example["pixel_values"]) for example in batch]
        pixel_values = torch.tensor(np.stack(pixel_list))

        # Pad labels to max length in batch and replace pad_token_id with -100
        label_tensors = [torch.tensor(example["labels"], dtype=torch.long) for example in batch]
        labels_padded = torch.nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        labels_padded = labels_padded.masked_fill(labels_padded == processor.tokenizer.pad_token_id, -100)

        return {"pixel_values": pixel_values, "labels": labels_padded}


if __name__ == "__main__":
    # Decide to use trained model for evaluation or not
    use_trained = input("Use trained model for evaluation? (y/n): ").lower() == 'y'

    if use_trained:
        fold = input("Enter fold of dataset to evaluate (e.g., '0' for fold-0): ")
        fold = int(fold)

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./models/trocr-hasyv2-{fold}",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            predict_with_generate=True,
            num_train_epochs=5,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=torch.cuda.is_available(),
        )
        model = VisionEncoderDecoderModel.from_pretrained(f"./trocr-hasyv2{fold}")
        processor = TrOCRProcessor.from_pretrained(f"./trocr-hasyv2{fold}")
        data_files = {"train": "train.csv", "test": "test.csv"}
        dataset_name = './hasyv2_ds/classification-task/fold-' + str(fold) + '/'
        dataset = load_dataset(dataset_name, data_files=data_files)
        test_dataset = dataset["test"].map(preprocess_function, batched=True)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=custom_data_collator,
        )
        # Compute sequence-level exact-match accuracy using the helper
        accuracy_metric(trainer, test_dataset)
        exit(0)

    # Test Torch CUDA availability
    print("Torch CUDA available:", torch.cuda.is_available())
    out = input("Proceed with training? (y/n): ")
    if out.lower() != 'y':
        exit(0)
    #make sure the input is number between 0-10
    ext = input("Enter fold of dataset to train (e.g., '0' for fold-0): ")
    ext = int(ext)
    while not (0 <= ext <= 10):
        ext = int(input("Invalid input. Please enter a number between 0 and 10: "))

    # 1) Load model & processor
    model_id = "microsoft/trocr-small-stage1"
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Ensure the model config has necessary decoder/generation tokens set. Some checkpoints
    # don't set these values which leads to errors during generation/training.
    # Prefer bos_token_id, then cls_token_id, then eos as the decoder start token.
    if model.config.decoder_start_token_id is None:
        if getattr(processor.tokenizer, "bos_token_id", None) is not None:
            model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
        elif getattr(processor.tokenizer, "cls_token_id", None) is not None:
            model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        else:
            # fallback to eos_token_id if nothing else is available
            model.config.decoder_start_token_id = getattr(processor.tokenizer, "eos_token_id", None)

    # Set pad / eos token ids from tokenizer when available
    if getattr(processor.tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if getattr(processor.tokenizer, "eos_token_id", None) is not None:
        model.config.eos_token_id = processor.tokenizer.eos_token_id

    # fold-0 for testing, change as needed
    data_files = {"train": "train.csv", "test": "test.csv"}
    dataset_name = './hasyv2_ds/classification-task/fold-' + str(ext) + '/'
    dataset = load_dataset(dataset_name, data_files=data_files)


    train_dataset = dataset["train"].map(preprocess_function, batched=True)
    test_dataset = dataset["test"].map(preprocess_function, batched=True)

    # 3) Define training arguments and trainer

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./trocr-hasyv2-{ext}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        num_train_epochs=10,
        learning_rate=3e-5,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    data_collator = custom_data_collator
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()


    # Save the model and processor

    trainer.save_model(f"./models/trocr-hasyv2-{ext}")
    processor.save_pretrained(f"./models/trocr-hasyv2-{ext}")

    # 4) Evaluation
    metrics = trainer.evaluate()
    print(metrics)

    accuracy_metric(trainer, test_dataset)
