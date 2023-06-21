# Data
import json
# NEE
import spacy
from spacy.lang.en import English

def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def generate_better_names(file):
    data = load_data(file)
    updated_names = []
    # Remove unneeded words from the company name
    words = [
        "Company",
        "Corporation ",
        "Inc.",
        "Inc",
        "Technology",
        "Common Stock",
        "Group",
    ]
    
    for item in data:
        itemList = []
        itemList.append(item)

        # Some companies are called and mentioned with only its first name
        item1 = item
        item1 = item1.replace("The", "")
        if item1[-1] == " ":
            item1 = item1[:-1]
        itemList.append(item1.split(" ")[0])

        for word in words:
            item = item.split(word, 1)[0]
        # item = re.sub("", "", item)
        if len(item) != 0:
            if item[-1] == " ":
                item = item[:-1]
            itemList.append(item)
            itemList = list(set(itemList))
            updated_names.append(itemList)

    return updated_names


def create_training_data(file, typeo):
    opened_file = open(file)
    data = json.load(opened_file)
    patterns = []
    print(len(data))
    for index, itemList in enumerate(data):
        for item in itemList:
            # print(item, typeo[index])
            pattern = {"label": typeo[index], "pattern": item}
            patterns.append(pattern)
    for item in typeo:
        pattern = {"label": item, "pattern": item}

        patterns.append(pattern)
    return patterns


def generate_rules(patterns):
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    nlp.to_disk("H:/My Drive/GP/data/trained_files/trained_ner")


def test_model(text):
    nlp = spacy.load("H:/My Drive/GP/data/trained_files/trained_ner")
    doc = nlp(text)
    results = []
    for ent in doc.ents:
        results.append(ent.label_)
    return list((set(results)))