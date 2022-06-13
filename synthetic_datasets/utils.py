import json
import random


def fill_in_template(template: str, object1: str, object2: str):
    return template.replace("$1", object1).replace("$2", object2)


def create_text_jsonl(fp, text_data, min_size=None):
    if isinstance(text_data, str):
        if min_size is not None:
            text_data = [text_data] * min_size
        with open(fp, "w") as f:
            f.write(json.dumps({"text": text_data}) + "\n")
    elif isinstance(text_data, list):
        if min_size is not None:
            text_data = text_data * ((min_size + 1) // len(text_data))
        with open(fp, "w") as f:
            for text in text_data:
                f.write(json.dumps({"text": text}) + "\n")
    else:
        raise NotImplementedError(f"object {text_data} not supported")


def integrate_within_existing_data(fp, sentence, existing_data_path, seed=0):
    random.seed(seed)
    with open(existing_data_path, 'r') as existing_data:
        data = [(random.random(), line) for line in existing_data]
    data.sort()
    with open(fp, 'w') as f:
        f.write(json.dumps({"text": sentence}) + "\n")
        for _, line in data:
            f.write(line)
