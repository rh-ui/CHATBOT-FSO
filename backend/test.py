import json
import csv
from collections import defaultdict

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepare_samples(data):
    samples = []
    intents = set()

    for entry in data:
        intent = entry.get("intent")
        intents.add(intent)
        questions = entry.get("question", {})
        # questions is dict of language -> list[str]
        for lang, q_list in questions.items():
            for q in q_list:
                q_clean = q.strip()
                if q_clean:
                    samples.append((q_clean, intent))

    return samples, sorted(list(intents))

def save_csv(samples, path):
    with open(path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["question", "intent"])
        writer.writerows(samples)

def save_intent2id(intents, path):
    intent2id = {intent: idx for idx, intent in enumerate(intents)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(intent2id, f, indent=2, ensure_ascii=False)
    return intent2id

def main():
    input_json = "dataset.json"  # Your full dataset file
    output_csv = "train_data.csv"
    output_intent2id = "intent2id.json"

    data = load_dataset(input_json)
    samples, intents = prepare_samples(data)
    print(f"Total samples: {len(samples)}")
    print(f"Total intents: {len(intents)}")

    save_csv(samples, output_csv)
    save_intent2id(intents, output_intent2id)

if __name__ == "__main__":
    main()
