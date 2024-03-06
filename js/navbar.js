from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

def generate_paragraph(car_info, min_word_count=50):
    # Construct text from car info
    text = f"This {car_info['year']} {car_info['make']} {car_info['type']} is in {car_info['condition']} condition. It comes in {car_info['color']} color with a {car_info['trim']} trim."

    # Add [CLS] and [SEP] tokens
    text = "[CLS] " + text + " [SEP]"
    
    generated_text = ""
    
    while len(tokenizer.tokenize(generated_text)) < min_word_count:
        # Tokenize input text
        inputs = tokenizer.encode_plus(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

        # Find the index of [SEP] token
        sep_index = torch.where(inputs['input_ids'] == tokenizer.sep_token_id)

        # If [SEP] token is not present, insert one at the end of the input_ids
        if sep_index[0].size(0) == 0:
            inputs['input_ids'][0][-1] = tokenizer.sep_token_id
            sep_index = (torch.tensor([0]), torch.tensor([inputs['input_ids'].size(1) - 1]))

        # Predict missing words
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        for i in range(sep_index[1].size(0)):
            masked_index = sep_index[1][i]
            predicted_index = torch.argmax(predictions[0, masked_index]).item()
            predicted_token = tokenizer.decode([predicted_index])
            inputs['input_ids'][0][masked_index] = predicted_index

        # Generate the full paragraph
        generated_text = tokenizer.decode(inputs['input_ids'][0])

    return generated_text

# Example JSON inputimport torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the list of parameters to initialize
parameters_to_initialize = ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']

# Load pre-trained BERT model configuration
config = BertConfig.from_pretrained('bert-base-uncased')

# Initialize the BERT model
model = BertForMaskedLM(config)

# Initialize only the specified parameters
for name, param in model.named_parameters():
    if any(param_name in name for param_name in parameters_to_initialize):
        param.requires_grad = True
        if 'weight' in name:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0.0)

def generate_paragraph(car_info, min_word_count=50):
    # Construct text from car info
    text = f"This {car_info['year']} {car_info['make']} {car_info['type']} is in {car_info['condition']} condition. It comes in {car_info['color']} color with a {car_info['trim']} trim."

    # Add [CLS] and [SEP] tokens
    text = "[CLS] " + text + " [SEP]"

    generated_text = ""

    while len(tokenizer.tokenize(generated_text)) < min_word_count:
        # Tokenize input text
        inputs = tokenizer.encode_plus(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

        # Find the index of [SEP] token
        sep_index = torch.where(inputs['input_ids'] == tokenizer.sep_token_id)

        # If [SEP] token is not present, insert one at the end of the input_ids
        if sep_index[0].size(0) == 0:
            inputs['input_ids'][0][-1] = tokenizer.sep_token_id
            sep_index = (torch.tensor([0]), torch.tensor([inputs['input_ids'].size(1) - 1]))

        # Predict missing words
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        for i in range(sep_index[1].size(0)):
            masked_index = sep_index[1][i]
            predicted_index = torch.argmax(predictions[0, masked_index]).item()
            predicted_token = tokenizer.decode([predicted_index])
            inputs['input_ids'][0][masked_index] = predicted_index

        # Generate the full paragraph
        generated_text = tokenizer.decode(inputs['input_ids'][0])

    return generated_text

# Example JSON input
car_info = {
    "make": "Toyota",
    "type": "sedan",
    "year": 2015,
    "trim": "LE",
    "color": "blue",
    "condition": "excellent"
}

# Generate paragraph
paragraph = generate_paragraph(car_info, min_word_count=50)

print("Generated paragraph:")
print(paragraph)
z