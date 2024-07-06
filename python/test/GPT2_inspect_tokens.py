from transformers import GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Special token used by GPT-2
start_token = "<|endoftext|>" #50256


# Encode the start token
input_ids = tokenizer.encode(start_token, return_tensors="pt")


# Decode the encoded input IDs back to text
decoded_terms = tokenizer.decode(input_ids[0])

print(f"Encoded Input IDs: {input_ids}")
print(f"Decoded Terms: {decoded_terms}")



##############################################################################
#text = "Tell me a haiku"
text = "Hi. How are you?"
# Encode the start token
input_ids = tokenizer.encode(text, return_tensors="pt")

# Decode the encoded input IDs back to text
decoded_terms = tokenizer.decode(input_ids[0])

print(f"Encoded Input IDs: {input_ids}")
print(f"Decoded Terms: {decoded_terms}")

response = torch.tensor([387, 13, 198, 11, 410, 393, 345, 338, 11, 11, 198, 11, 13]).long()
response = torch.tensor([15902, 11, 198, 314, 13, 357, 46, 11, 198, 11, 198, 7, 11, 198]).long()

# Decode the encoded input IDs back to text
decoded_terms = tokenizer.decode(response)
print(f"Decoded Terms: {decoded_terms}")








