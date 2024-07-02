#"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare initial inputs
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Perform the initial model run to get past_key_values
outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
logits = outputs.logits
print(logits.shape)
past_key_values = outputs.past_key_values

print("Initial Logits:", logits.shape)
print("Initial Past Key Values Shapes:", [pkv[0].shape for pkv in past_key_values])

# Prepare inputs for the second run
new_input_text = " I'm doing well, thank you."
new_inputs = tokenizer(new_input_text, return_tensors='pt')
new_input_ids = new_inputs['input_ids']

# Extend the attention mask to account for past key values
extended_attention_mask = torch.cat(
    [attention_mask, torch.ones((attention_mask.size(0), new_input_ids.size(-1)), dtype=attention_mask.dtype)], dim=-1
)

# Run the model with past_key_values and extended attention mask
outputs = model(new_input_ids, attention_mask=extended_attention_mask, past_key_values=past_key_values, use_cache=True)
logits = outputs.logits
print(logits.shape)
past_key_values = outputs.past_key_values

print("Logits:", logits.shape)
print("Past Key Values:", len(past_key_values))
for pkv in past_key_values:
    print("\t pkv:", len(pkv))
    for pkvi in pkv:
        print("\t", pkvi.shape)


class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPT2Wrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, past_key_values):
        print(past_key_values.sum())
        #if torch.any(past_key_values):
        if torch.abs(past_key_values.sum()) > 0:
            # Reshape past_key_values from (num_layers * 2, batch_size, num_heads, seq_length, head_dim)
            num_layers = past_key_values.shape[0] // 2
            past_key_values = tuple(
                (past_key_values[i], past_key_values[i + num_layers])
                for i in range(num_layers)
            )
            outputs = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:,-1,:]

        # Flatten past_key_values to a single tensor for ONNX export
        past_key_values_flat = torch.cat(
            [torch.cat([pk[0].unsqueeze(0), pk[1].unsqueeze(0)], dim=0) for pk in outputs.past_key_values], dim=0)
        return logits, past_key_values_flat

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


# Initialize the wrapper
wrapper = GPT2Wrapper(model)
wrapper.freeze_parameters()

with torch.no_grad():
    logits, past_key_values = wrapper(input_ids, attention_mask, torch.zeros(2,2))
    print("First pass:")
    print("input_ids:", input_ids.shape)
    print("attention_mask:", attention_mask.shape)
    print("past_key_values:", past_key_values.shape)

    logits, past_key_values_2 = wrapper(new_input_ids, extended_attention_mask, past_key_values)
    print("Second pass:")
    print("new_input_ids:", new_input_ids.shape)
    print("extended_attention_mask:", extended_attention_mask.shape)
    print("past_key_values:", past_key_values.shape)
    print("past_key_values_2:", past_key_values_2.shape)

# Export to ONNX
torch.onnx.export(
    wrapper,
    (new_input_ids, extended_attention_mask, past_key_values),
    "onnx_model/gpt2_with_kv_in.onnx",
    input_names=["input_ids", "attention_mask", "past_key_values_input"],
    output_names=["logits", "past_key_values_output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "new_sequence"},
        "attention_mask": {0: "batch_size", 1: "new_and_last_sequence"},
        "logits": {0: "batch_size"}, #, 1: "sequence"},
        "past_key_values_input": {1: "batch_size", 3: "last_sequence"},
        "past_key_values_output": {1: "batch_size", 3: "new_and_last_sequence"}
        #"past_key_values_input": {1: "batch_size", 2: "num_heads", 3: "sequence", 4: "head_dim"},
        #"past_key_values_output": {1: "batch_size", 2: "num_heads", 3: "sequence", 4: "head_dim"}
    },
    opset_version=11
)

import onnx
import onnxruntime as ort

# Verify the exported ONNX model
onnx_model = onnx.load("onnx_model/gpt2_with_kv_in.onnx")
onnx.checker.check_model(onnx_model)

# Initialize ONNX Runtime session
ort_session = ort.InferenceSession("onnx_model/gpt2_with_kv_in.onnx")

# Prepare dummy inputs for ONNX
new_input_ids_ort = new_input_ids.numpy()
extended_attention_mask_ort = extended_attention_mask.numpy()
past_key_values_ort = past_key_values.numpy()

# Create input feed dictionary
onnx_inputs = {
    "input_ids": new_input_ids_ort,
    "attention_mask": extended_attention_mask_ort,
    "past_key_values_input": past_key_values_ort
}

# Run inference
outputs = ort_session.run(None, onnx_inputs)

# Verify the outputs
logits_ort = outputs[0]
past_key_values_ort = outputs[1]
print(f"Logits: {logits_ort.shape}")
print(f"Past Key Values: {past_key_values_ort.shape}")

#"""

import torch
import onnx
import onnxruntime as ort
# Dummy inputs for tracing
batch_size = 1
seq_length = 2
num_heads = 12
head_dim = 64
num_layers = 12

input_ids = torch.randint(0, 50257, (batch_size, seq_length)).long()   # Example input_ids
attention_mask = torch.ones((batch_size, seq_length)).long()         # Example attention mask
past_key_values = torch.zeros((num_layers * 2, batch_size, num_heads, seq_length, head_dim))

new_input_ids = torch.randint(0, 50257, (batch_size, seq_length)).long()  # Example input_ids
# Extend the attention mask to account for past key values
extended_attention_mask = torch.cat(
    [attention_mask, torch.ones((attention_mask.size(0), new_input_ids.size(-1)), dtype=attention_mask.dtype)], dim=-1
)

# Verify the exported ONNX model
onnx_model = onnx.load("onnx_model/gpt2_with_kv_in.onnx")
onnx.checker.check_model(onnx_model)

# Initialize ONNX Runtime session
ort_session = ort.InferenceSession("onnx_model/gpt2_with_kv_in.onnx")

# Prepare dummy inputs for ONNX
new_input_ids_ort = new_input_ids.numpy()
extended_attention_mask_ort = extended_attention_mask.numpy()
past_key_values_ort = past_key_values.numpy()

# Create input feed dictionary
onnx_inputs = {
    "input_ids": new_input_ids_ort,
    "attention_mask": extended_attention_mask_ort,
    "past_key_values_input": past_key_values_ort
}


# Run inference
outputs = ort_session.run(None, onnx_inputs)

# Verify the outputs
logits_ort = outputs[0]
past_key_values_ort = outputs[1]
print(f"Logits: {logits_ort.shape}")
print(f"Past Key Values: {past_key_values_ort.shape}")



#import numpy as np
past_key_values_zero = torch.zeros((num_layers * 2, batch_size, num_heads, seq_length, head_dim))
onnx_inputs = {
    "input_ids": new_input_ids_ort,
    "attention_mask": extended_attention_mask_ort,
    "past_key_values_input": past_key_values_zero.numpy()
}
# Run inference
outputs = ort_session.run(None, onnx_inputs)

# Verify the outputs
logits_ort = outputs[0]
past_key_values_ort = outputs[1]
print(f"Logits: {logits_ort.shape}")
print(f"Past Key Values: {past_key_values_ort.shape}")

#"""