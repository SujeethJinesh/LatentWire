#!/usr/bin/env python
# telepathy/eval_text_relay_baseline.py
"""
Text-Relay Baseline: Llama summarizes → text → Mistral classifies

This baseline answers: "Is the bridge benefit from:
  (a) Llama's encoding being better than Mistral's reading? OR
  (b) Something special about the latent space?"

If text-relay matches bridge performance, the benefit is from (a).
If bridge >> text-relay, there's something special about latent transfer.
"""
import os
import torch
import json
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def generate_summary(model, tokenizer, text, max_input_len=512, max_summary_len=50):
    """Have Llama generate a short summary of the input."""
    prompt = f"Summarize this in one sentence:\n\n{text[:max_input_len]}\n\nSummary:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len + 50)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_summary_len,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    summary = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return summary.strip()


def classify_with_mistral(model, tokenizer, summary, task_prompt):
    """Have Mistral classify based on Llama's summary."""
    prompt = f"{task_prompt}\n\nText: {summary}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip().lower()


def eval_sst2(llama_model, llama_tok, mistral_model, mistral_tok, num_samples=200):
    """Evaluate text-relay on SST-2."""
    print("\n" + "=" * 60)
    print("SST-2 TEXT-RELAY BASELINE")
    print("=" * 60)

    dataset = load_dataset("glue", "sst2", split="validation")

    task_prompt = "Is the sentiment positive or negative?"

    correct = 0
    total = 0

    for i in tqdm(range(min(num_samples, len(dataset))), desc="SST-2"):
        item = dataset[i]
        text = item['sentence']
        label = "positive" if item['label'] == 1 else "negative"

        # Step 1: Llama summarizes
        summary = generate_summary(llama_model, llama_tok, text)

        # Step 2: Mistral classifies from summary
        prediction = classify_with_mistral(mistral_model, mistral_tok, summary, task_prompt)

        if label in prediction:
            correct += 1
        total += 1

        if i < 3:
            print(f"\n[{i}] Original: {text[:60]}...")
            print(f"    Summary: {summary[:60]}...")
            print(f"    Label: {label}, Pred: {prediction[:20]}")

    accuracy = 100 * correct / total
    print(f"\nSST-2 Text-Relay Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return {"accuracy": accuracy, "correct": correct, "total": total}


def eval_agnews(llama_model, llama_tok, mistral_model, mistral_tok, num_samples=200):
    """Evaluate text-relay on AG News."""
    print("\n" + "=" * 60)
    print("AG NEWS TEXT-RELAY BASELINE")
    print("=" * 60)

    dataset = load_dataset("ag_news", split="test")
    labels = ["world", "sports", "business", "science"]

    task_prompt = "What is the topic: world, sports, business, or science?"

    correct = 0
    total = 0

    # Permissive matching for science
    science_synonyms = ["science", "technology", "tech", "sci/tech", "scitech"]

    for i in tqdm(range(min(num_samples, len(dataset))), desc="AG News"):
        item = dataset[i]
        text = item['text']
        label = labels[item['label']]

        # Step 1: Llama summarizes
        summary = generate_summary(llama_model, llama_tok, text[:500])

        # Step 2: Mistral classifies from summary
        prediction = classify_with_mistral(mistral_model, mistral_tok, summary, task_prompt)

        # Check match (permissive for science)
        if label == "science":
            is_correct = any(syn in prediction for syn in science_synonyms)
        else:
            is_correct = label in prediction

        if is_correct:
            correct += 1
        total += 1

        if i < 3:
            print(f"\n[{i}] Original: {text[:60]}...")
            print(f"    Summary: {summary[:60]}...")
            print(f"    Label: {label}, Pred: {prediction[:20]}")

    accuracy = 100 * correct / total
    print(f"\nAG News Text-Relay Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return {"accuracy": accuracy, "correct": correct, "total": total}


def eval_trec_text_baseline(model, tokenizer, num_samples=200):
    """Evaluate direct text classification on TREC 6-class question type."""
    print("\n" + "=" * 60)
    print("TREC TEXT BASELINE (6-class)")
    print("=" * 60)

    dataset = load_dataset("CogComp/trec", trust_remote_code=True)
    labels = dataset['train'].features['coarse_label'].names
    test_data = dataset['test']

    print(f"Classes: {len(labels)} - {labels}")
    print(f"Test samples: {len(test_data)}")

    correct = 0
    total = 0
    samples = []

    for i in tqdm(range(min(num_samples, len(test_data))), desc="TREC"):
        item = test_data[i]
        text = item['text']
        label = labels[item['coarse_label']]

        # Direct classification prompt
        prompt = f"""Classify this question into one of these types: ABBR (abbreviation), DESC (description), ENTY (entity), HUM (human), LOC (location), NUM (numeric).

Question: {text}

Type:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        prediction = response.strip().lower()

        # Check if label matches
        is_correct = label.lower() in prediction

        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            samples.append(f"Q: {text[:50]}... | GT: {label} | Pred: {prediction[:20]}")

    accuracy = 100 * correct / total
    print(f"\nTREC Text Baseline Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("\nSamples:")
    for s in samples:
        print(f"  {s}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "num_classes": len(labels)}


def eval_trec_text_relay(llama_model, llama_tok, mistral_model, mistral_tok, num_samples=200):
    """Evaluate text-relay on TREC: Llama summarizes → text → Mistral classifies."""
    print("\n" + "=" * 60)
    print("TREC TEXT-RELAY BASELINE (6-class)")
    print("=" * 60)
    print("Pipeline: Llama summarizes question → text → Mistral classifies type")
    print("=" * 60)

    dataset = load_dataset("CogComp/trec", trust_remote_code=True)
    labels = dataset['train'].features['coarse_label'].names
    test_data = dataset['test']

    print(f"Classes: {len(labels)} - {labels}")
    print(f"Test samples: {len(test_data)}")

    # Task prompt for Mistral classification
    task_prompt = "What type of question is this: ABBR (abbreviation), DESC (description), ENTY (entity), HUM (human), LOC (location), or NUM (numeric)? Respond with just the type."

    correct = 0
    total = 0
    samples = []

    for i in tqdm(range(min(num_samples, len(test_data))), desc="TREC Relay"):
        item = test_data[i]
        text = item['text']
        label = labels[item['coarse_label']]

        # Step 1: Llama summarizes/rephrases the question
        summary = generate_summary(llama_model, llama_tok, text, max_input_len=128, max_summary_len=40)

        # Step 2: Mistral classifies from Llama's summary
        prediction = classify_with_mistral(mistral_model, mistral_tok, summary, task_prompt)

        # Check if label matches
        is_correct = label.lower() in prediction

        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            samples.append({
                "question": text[:50],
                "summary": summary[:50],
                "label": label,
                "prediction": prediction[:20]
            })

    accuracy = 100 * correct / total
    print(f"\nTREC Text-Relay Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("\nSamples:")
    for s in samples:
        print(f"  Q: {s['question']}...")
        print(f"  Summary: {s['summary']}...")
        print(f"  GT: {s['label']} | Pred: {s['prediction']}")
        print()

    return {"accuracy": accuracy, "correct": correct, "total": total, "num_classes": len(labels)}


def eval_sst2_text_baseline(model, tokenizer, num_samples=200):
    """Evaluate direct text classification on SST-2 (no relay, just model + text)."""
    print("\n" + "=" * 60)
    print("SST-2 TEXT BASELINE")
    print("=" * 60)

    dataset = load_dataset("glue", "sst2", split="validation")

    correct = 0
    total = 0
    samples = []

    for i in tqdm(range(min(num_samples, len(dataset))), desc="SST-2"):
        item = dataset[i]
        text = item['sentence']
        label = "positive" if item['label'] == 1 else "negative"

        # Direct classification prompt
        prompt = f"""Is the sentiment of this text positive or negative?

Text: {text}

Sentiment:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        prediction = response.strip().lower()

        # Check if label matches
        is_correct = label in prediction

        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            samples.append(f"Text: {text[:50]}... | GT: {label} | Pred: {prediction[:20]}")

    accuracy = 100 * correct / total
    print(f"\nSST-2 Text Baseline Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("\nSamples:")
    for s in samples:
        print(f"  {s}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "num_classes": 2}


def eval_agnews_text_baseline(model, tokenizer, num_samples=200):
    """Evaluate direct text classification on AG News (no relay, just model + text)."""
    print("\n" + "=" * 60)
    print("AG NEWS TEXT BASELINE")
    print("=" * 60)

    dataset = load_dataset("ag_news", split="test")
    labels = ["world", "sports", "business", "science"]

    correct = 0
    total = 0
    samples = []

    # Permissive matching for science
    science_synonyms = ["science", "technology", "tech", "sci/tech", "scitech"]

    for i in tqdm(range(min(num_samples, len(dataset))), desc="AG News"):
        item = dataset[i]
        text = item['text']
        label = labels[item['label']]

        # Direct classification prompt
        prompt = f"""What is the topic of this article: world, sports, business, or science?

Text: {text[:500]}

Topic:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        prediction = response.strip().lower()

        # Check match (permissive for science)
        if label == "science":
            is_correct = any(syn in prediction for syn in science_synonyms)
        else:
            is_correct = label in prediction

        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            samples.append(f"Text: {text[:50]}... | GT: {label} | Pred: {prediction[:20]}")

    accuracy = 100 * correct / total
    print(f"\nAG News Text Baseline Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("\nSamples:")
    for s in samples:
        print(f"  {s}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "num_classes": 4}


def eval_banking77_text_baseline(model, tokenizer, num_samples=200):
    """Evaluate direct text classification on Banking77 (no relay, just model + text)."""
    print("\n" + "=" * 60)
    print("BANKING77 TEXT BASELINE")
    print("=" * 60)

    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    labels = dataset['train'].features['label'].names
    test_data = dataset['test']

    print(f"Classes: {len(labels)}")
    print(f"Test samples: {len(test_data)}")

    # Create a prompt that lists some example intents
    label_examples = ", ".join(labels[:10]) + "..."

    correct = 0
    total = 0
    samples = []

    for i in tqdm(range(min(num_samples, len(test_data))), desc="Banking77"):
        item = test_data[i]
        text = item['text']
        label = labels[item['label']]

        # Direct classification prompt
        prompt = f"""Classify this banking query into one of the intents.

Query: {text}

Intent:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        prediction = response.strip().lower()

        # Check if label matches (flexible matching)
        label_lower = label.lower().replace("_", " ")
        is_correct = (label_lower in prediction) or (label.lower() in prediction)

        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            samples.append(f"Q: {text[:50]}... | GT: {label} | Pred: {prediction[:30]}")

    accuracy = 100 * correct / total
    print(f"\nBanking77 Text Baseline Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("\nSamples:")
    for s in samples:
        print(f"  {s}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "num_classes": len(labels)}


def eval_banking77_text_relay(llama_model, llama_tok, mistral_model, mistral_tok, num_samples=200):
    """Evaluate text-relay on Banking77: Llama summarizes → text → Mistral classifies."""
    print("\n" + "=" * 60)
    print("BANKING77 TEXT-RELAY BASELINE")
    print("=" * 60)
    print("Pipeline: Llama summarizes query → text → Mistral classifies intent")
    print("=" * 60)

    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    labels = dataset['train'].features['label'].names
    test_data = dataset['test']

    print(f"Classes: {len(labels)}")
    print(f"Test samples: {len(test_data)}")

    # Task prompt for Mistral classification
    task_prompt = "What is the banking intent of this query? Respond with just the intent category."

    correct = 0
    total = 0
    samples = []

    for i in tqdm(range(min(num_samples, len(test_data))), desc="Banking77 Relay"):
        item = test_data[i]
        text = item['text']
        label = labels[item['label']]

        # Step 1: Llama summarizes/rephrases the query
        summary = generate_summary(llama_model, llama_tok, text, max_input_len=256, max_summary_len=60)

        # Step 2: Mistral classifies from Llama's summary
        prediction = classify_with_mistral(mistral_model, mistral_tok, summary, task_prompt)

        # Check if label matches (flexible matching)
        label_lower = label.lower().replace("_", " ")
        is_correct = (label_lower in prediction) or (label.lower() in prediction)

        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            samples.append({
                "query": text[:50],
                "summary": summary[:50],
                "label": label,
                "prediction": prediction[:30]
            })

    accuracy = 100 * correct / total
    print(f"\nBanking77 Text-Relay Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("\nSamples:")
    for s in samples:
        print(f"  Q: {s['query']}...")
        print(f"  Summary: {s['summary']}...")
        print(f"  GT: {s['label']} | Pred: {s['prediction']}")
        print()

    return {"accuracy": accuracy, "correct": correct, "total": total, "num_classes": len(labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/text_relay_baseline")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--llama_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--mistral_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use (both models fit on one 80GB GPU)")
    parser.add_argument("--banking77", action="store_true", help="Run Banking77 text baseline (direct classification)")
    parser.add_argument("--banking77_relay", action="store_true", help="Run Banking77 text-relay (Llama summarizes → Mistral classifies)")
    parser.add_argument("--trec", action="store_true", help="Run TREC text baseline (direct classification)")
    parser.add_argument("--trec_relay", action="store_true", help="Run TREC text-relay (Llama summarizes → Mistral classifies)")
    parser.add_argument("--sst2_text", action="store_true", help="Run SST-2 text baseline (direct classification, both models)")
    parser.add_argument("--agnews_text", action="store_true", help="Run AG News text baseline (direct classification, both models)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Single GPU mode - both models fit on one 80GB H100
    num_gpus = torch.cuda.device_count()
    device = torch.device(f"cuda:{args.gpu}") if num_gpus > args.gpu else torch.device("cpu")
    print(f"Using GPU {args.gpu} ({num_gpus} available): {device}")

    if args.banking77:
        # Banking77 text baseline mode - evaluate both models directly
        print("=" * 60)
        print("BANKING77 TEXT BASELINES")
        print("=" * 60)
        print("Evaluating direct text classification (no bridge)")
        print("=" * 60)

        results = {
            "experiment": "banking77_text_baselines",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "llama_model": args.llama_model,
                "mistral_model": args.mistral_model,
                "num_samples": args.num_samples,
            },
            "results": {}
        }

        # Load and eval Mistral
        print("\nLoading Mistral...")
        mistral_tok = AutoTokenizer.from_pretrained(args.mistral_model)
        mistral_tok.pad_token = mistral_tok.eos_token
        mistral_model = AutoModelForCausalLM.from_pretrained(
            args.mistral_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        mistral_model.eval()

        results["results"]["mistral"] = eval_banking77_text_baseline(
            mistral_model, mistral_tok, args.num_samples
        )

        # Free Mistral memory
        del mistral_model
        torch.cuda.empty_cache()

        # Load and eval Llama
        print("\nLoading Llama...")
        llama_tok = AutoTokenizer.from_pretrained(args.llama_model)
        llama_tok.pad_token = llama_tok.eos_token
        llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        llama_model.eval()

        results["results"]["llama"] = eval_banking77_text_baseline(
            llama_model, llama_tok, args.num_samples
        )

        # Summary
        print("\n" + "=" * 60)
        print("BANKING77 TEXT BASELINES SUMMARY")
        print("=" * 60)
        print(f"Mistral Text: {results['results']['mistral']['accuracy']:.1f}%")
        print(f"Llama Text:   {results['results']['llama']['accuracy']:.1f}%")
        print(f"Bridge (16 tokens): 21.5%")
        print(f"Random: 1.3%")
        print("=" * 60)

        # Save results
        json_path = os.path.join(args.output_dir, "banking77_baselines.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    elif args.banking77_relay:
        # Banking77 text-relay mode: Llama summarizes → Mistral classifies
        print("=" * 60)
        print("BANKING77 TEXT-RELAY BASELINE")
        print("=" * 60)
        print("Pipeline: Llama summarizes → text → Mistral classifies")
        print(f"Llama: {args.llama_model}")
        print(f"Mistral: {args.mistral_model}")
        print("=" * 60)

        # Load both models (need both for relay)
        print("\nLoading Llama...")
        llama_tok = AutoTokenizer.from_pretrained(args.llama_model)
        llama_tok.pad_token = llama_tok.eos_token
        llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        llama_model.eval()

        print("Loading Mistral...")
        mistral_tok = AutoTokenizer.from_pretrained(args.mistral_model)
        mistral_tok.pad_token = mistral_tok.eos_token
        mistral_model = AutoModelForCausalLM.from_pretrained(
            args.mistral_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        mistral_model.eval()

        results = {
            "experiment": "banking77_text_relay",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "llama_model": args.llama_model,
                "mistral_model": args.mistral_model,
                "num_samples": args.num_samples,
            },
            "results": {}
        }

        # Run Banking77 text-relay evaluation
        results["results"]["banking77"] = eval_banking77_text_relay(
            llama_model, llama_tok, mistral_model, mistral_tok, args.num_samples
        )

        # Summary with comparison
        print("\n" + "=" * 60)
        print("BANKING77 COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Text-Relay (Llama→text→Mistral): {results['results']['banking77']['accuracy']:.1f}%")
        print(f"Bridge (16 tokens):              21.5%")
        print(f"Mistral Text (direct):           19.5%")
        print(f"Llama Text (direct):             22.0%")
        print(f"Random:                          1.3%")
        print("=" * 60)

        # Save results
        json_path = os.path.join(args.output_dir, "banking77_relay_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    elif args.trec:
        # TREC text baseline mode - evaluate both models directly
        print("=" * 60)
        print("TREC TEXT BASELINES (6-class)")
        print("=" * 60)
        print("Evaluating direct text classification (no bridge)")
        print("=" * 60)

        results = {
            "experiment": "trec_text_baselines",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "llama_model": args.llama_model,
                "mistral_model": args.mistral_model,
                "num_samples": args.num_samples,
            },
            "results": {}
        }

        # Load and eval Mistral
        print("\nLoading Mistral...")
        mistral_tok = AutoTokenizer.from_pretrained(args.mistral_model)
        mistral_tok.pad_token = mistral_tok.eos_token
        mistral_model = AutoModelForCausalLM.from_pretrained(
            args.mistral_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        mistral_model.eval()

        results["results"]["mistral"] = eval_trec_text_baseline(
            mistral_model, mistral_tok, args.num_samples
        )

        # Free Mistral memory
        del mistral_model
        torch.cuda.empty_cache()

        # Load and eval Llama
        print("\nLoading Llama...")
        llama_tok = AutoTokenizer.from_pretrained(args.llama_model)
        llama_tok.pad_token = llama_tok.eos_token
        llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        llama_model.eval()

        results["results"]["llama"] = eval_trec_text_baseline(
            llama_model, llama_tok, args.num_samples
        )

        # Summary
        print("\n" + "=" * 60)
        print("TREC TEXT BASELINES SUMMARY")
        print("=" * 60)
        print(f"Mistral Text: {results['results']['mistral']['accuracy']:.1f}%")
        print(f"Llama Text:   {results['results']['llama']['accuracy']:.1f}%")
        print(f"Bridge (16 tokens): 94.5%")
        print(f"Random (6 classes): 16.7%")
        print("=" * 60)

        # Save results
        json_path = os.path.join(args.output_dir, "trec_baselines.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    elif args.trec_relay:
        # TREC text-relay mode: Llama summarizes → Mistral classifies
        print("=" * 60)
        print("TREC TEXT-RELAY BASELINE (6-class)")
        print("=" * 60)
        print("Pipeline: Llama summarizes → text → Mistral classifies")
        print(f"Llama: {args.llama_model}")
        print(f"Mistral: {args.mistral_model}")
        print("=" * 60)

        # Load both models (need both for relay)
        print("\nLoading Llama...")
        llama_tok = AutoTokenizer.from_pretrained(args.llama_model)
        llama_tok.pad_token = llama_tok.eos_token
        llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        llama_model.eval()

        print("Loading Mistral...")
        mistral_tok = AutoTokenizer.from_pretrained(args.mistral_model)
        mistral_tok.pad_token = mistral_tok.eos_token
        mistral_model = AutoModelForCausalLM.from_pretrained(
            args.mistral_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        mistral_model.eval()

        results = {
            "experiment": "trec_text_relay",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "llama_model": args.llama_model,
                "mistral_model": args.mistral_model,
                "num_samples": args.num_samples,
            },
            "results": {}
        }

        # Run TREC text-relay evaluation
        results["results"]["trec"] = eval_trec_text_relay(
            llama_model, llama_tok, mistral_model, mistral_tok, args.num_samples
        )

        # Summary with comparison
        print("\n" + "=" * 60)
        print("TREC COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Text-Relay (Llama→text→Mistral): {results['results']['trec']['accuracy']:.1f}%")
        print(f"Bridge (16 tokens):              94.5%")
        print(f"Random (6 classes):              16.7%")
        print("=" * 60)

        # Save results
        json_path = os.path.join(args.output_dir, "trec_relay_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    elif args.sst2_text:
        # SST-2 text baseline mode - evaluate both models directly
        print("=" * 60)
        print("SST-2 TEXT BASELINES")
        print("=" * 60)
        print("Evaluating direct text classification (no bridge)")
        print("=" * 60)

        results = {
            "experiment": "sst2_text_baselines",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "llama_model": args.llama_model,
                "mistral_model": args.mistral_model,
                "num_samples": args.num_samples,
            },
            "results": {}
        }

        # Load and eval Mistral
        print("\nLoading Mistral...")
        mistral_tok = AutoTokenizer.from_pretrained(args.mistral_model)
        mistral_tok.pad_token = mistral_tok.eos_token
        mistral_model = AutoModelForCausalLM.from_pretrained(
            args.mistral_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        mistral_model.eval()

        results["results"]["mistral"] = eval_sst2_text_baseline(
            mistral_model, mistral_tok, args.num_samples
        )

        # Free Mistral memory
        del mistral_model
        torch.cuda.empty_cache()

        # Load and eval Llama
        print("\nLoading Llama...")
        llama_tok = AutoTokenizer.from_pretrained(args.llama_model)
        llama_tok.pad_token = llama_tok.eos_token
        llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        llama_model.eval()

        results["results"]["llama"] = eval_sst2_text_baseline(
            llama_model, llama_tok, args.num_samples
        )

        # Summary
        print("\n" + "=" * 60)
        print("SST-2 TEXT BASELINES SUMMARY")
        print("=" * 60)
        print(f"Mistral Text: {results['results']['mistral']['accuracy']:.1f}%")
        print(f"Llama Text:   {results['results']['llama']['accuracy']:.1f}%")
        print(f"Bridge (8 tokens): 94.7%")
        print(f"Text-Relay:   71.0%")
        print(f"Random:       50.0%")
        print("=" * 60)

        # Save results
        json_path = os.path.join(args.output_dir, "sst2_baselines.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    elif args.agnews_text:
        # AG News text baseline mode - evaluate both models directly
        print("=" * 60)
        print("AG NEWS TEXT BASELINES")
        print("=" * 60)
        print("Evaluating direct text classification (no bridge)")
        print("=" * 60)

        results = {
            "experiment": "agnews_text_baselines",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "llama_model": args.llama_model,
                "mistral_model": args.mistral_model,
                "num_samples": args.num_samples,
            },
            "results": {}
        }

        # Load and eval Mistral
        print("\nLoading Mistral...")
        mistral_tok = AutoTokenizer.from_pretrained(args.mistral_model)
        mistral_tok.pad_token = mistral_tok.eos_token
        mistral_model = AutoModelForCausalLM.from_pretrained(
            args.mistral_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        mistral_model.eval()

        results["results"]["mistral"] = eval_agnews_text_baseline(
            mistral_model, mistral_tok, args.num_samples
        )

        # Free Mistral memory
        del mistral_model
        torch.cuda.empty_cache()

        # Load and eval Llama
        print("\nLoading Llama...")
        llama_tok = AutoTokenizer.from_pretrained(args.llama_model)
        llama_tok.pad_token = llama_tok.eos_token
        llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        llama_model.eval()

        results["results"]["llama"] = eval_agnews_text_baseline(
            llama_model, llama_tok, args.num_samples
        )

        # Summary
        print("\n" + "=" * 60)
        print("AG NEWS TEXT BASELINES SUMMARY")
        print("=" * 60)
        print(f"Mistral Text: {results['results']['mistral']['accuracy']:.1f}%")
        print(f"Llama Text:   {results['results']['llama']['accuracy']:.1f}%")
        print(f"Bridge (8 tokens): 88.9%")
        print(f"Text-Relay:   64.5%")
        print(f"Random:       25.0%")
        print("=" * 60)

        # Save results
        json_path = os.path.join(args.output_dir, "agnews_baselines.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    else:
        # Original text-relay mode (SST-2 and AG News)
        print("=" * 60)
        print("TEXT-RELAY BASELINE")
        print("=" * 60)
        print("Pipeline: Llama summarizes → text → Mistral classifies")
        print(f"Llama: {args.llama_model}")
        print(f"Mistral: {args.mistral_model}")
        print("=" * 60)

        # Load Llama
        print("\nLoading Llama...")
        llama_tok = AutoTokenizer.from_pretrained(args.llama_model)
        llama_tok.pad_token = llama_tok.eos_token
        llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        llama_model.eval()

        # Load Mistral
        print("Loading Mistral...")
        mistral_tok = AutoTokenizer.from_pretrained(args.mistral_model)
        mistral_tok.pad_token = mistral_tok.eos_token
        mistral_model = AutoModelForCausalLM.from_pretrained(
            args.mistral_model,
            torch_dtype=torch.bfloat16
        ).to(device)
        mistral_model.eval()

        results = {
            "experiment": "text_relay_baseline",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "llama_model": args.llama_model,
                "mistral_model": args.mistral_model,
                "num_samples": args.num_samples,
            },
            "results": {}
        }

        # Run evaluations
        results["results"]["sst2"] = eval_sst2(
            llama_model, llama_tok, mistral_model, mistral_tok, args.num_samples
        )

        results["results"]["agnews"] = eval_agnews(
            llama_model, llama_tok, mistral_model, mistral_tok, args.num_samples
        )

        # Summary
        print("\n" + "=" * 60)
        print("TEXT-RELAY BASELINE SUMMARY")
        print("=" * 60)
        print(f"SST-2:   {results['results']['sst2']['accuracy']:.1f}%")
        print(f"AG News: {results['results']['agnews']['accuracy']:.1f}%")
        print("=" * 60)

        # Compare with bridge results
        print("\nComparison with Bridge:")
        print("| Task    | Text-Relay | Bridge | Mistral Text |")
        print("|---------|------------|--------|--------------|")
        print(f"| SST-2   | {results['results']['sst2']['accuracy']:.1f}%      | 94.7%  | 93.5%        |")
        print(f"| AG News | {results['results']['agnews']['accuracy']:.1f}%      | 88.9%  | 70.5%        |")

        # Save results
        json_path = os.path.join(args.output_dir, "text_relay_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
