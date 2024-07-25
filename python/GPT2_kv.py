#"""
import copy

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare initial inputs
input_text = "What is your favorite"
inputs = tokenizer(input_text, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Perform the initial model run to get past_key_values
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
logits = outputs.logits
print(logits.shape)
past_key_values = outputs.past_key_values


past_key_values_init = copy.deepcopy(past_key_values)

print("Initial Logits:", logits.shape)
print("Initial Past Key Values Shapes:", [pkv[0].shape for pkv in past_key_values])

# Prepare inputs for the second run
new_input_text = " sport?\n"
new_inputs = tokenizer(new_input_text, return_tensors='pt')
new_input_ids = new_inputs['input_ids']

# Extend the attention mask to account for past key values
extended_attention_mask = torch.cat(
    [attention_mask, torch.ones((attention_mask.size(0), new_input_ids.size(-1)), dtype=attention_mask.dtype)], dim=-1
)
extended_attention_mask_init = extended_attention_mask

# Run the model with past_key_values and extended attention mask
with torch.no_grad():
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



##################################################################################################
extended_attention_mask = copy.deepcopy(extended_attention_mask_init)
past_key_values = copy.deepcopy(past_key_values_init)


print(extended_attention_mask.shape)
print(new_input_ids.shape)
print(past_key_values[0][0].shape)
next_input = copy.deepcopy(new_input_ids)

output_ids = []
# Run the model with past_key_values and extended attention mask
for i in range(5):
    with torch.no_grad():
        outputs = model(next_input, attention_mask=extended_attention_mask, past_key_values=past_key_values, use_cache=True)
    current_out = torch.argmax(outputs.logits[:, -1, :]).item()
    output_ids.append( current_out )
    next_input = torch.tensor([[current_out]])
    past_key_values = outputs.past_key_values
    extended_attention_mask = torch.cat([ extended_attention_mask, torch.ones((1,1)) ], dim=1)

print('Output', output_ids)
print(logits.shape)
print(extended_attention_mask.shape)
################################


class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPT2Wrapper, self).__init__()
        self.model = model
        self.num_layers = 12

    def forward(self, input_ids, attention_mask, past_key_values):

        # Reshape past_key_values from (num_layers * 2, batch_size, num_heads, seq_length, head_dim)
        if torch.all(past_key_values == 0):
            past_key_values = None
        else:
            past_key_values = tuple(
                (past_key_values[i], past_key_values[i + 1])
                for i in range(0, 2 * self.num_layers, 2)
            )

        outputs = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        outputs_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        past_key_values_flat = torch.cat(
            [torch.stack(pk, dim=0) for pk in outputs.past_key_values], dim=0)
        return outputs_id, past_key_values_flat

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


# Initialize the wrapper
wrapper = GPT2Wrapper(model)
wrapper.freeze_parameters()

next_input = torch.cat([input_ids, new_input_ids], dim=1)
#past_key_values = torch.zeros( torch.Size([12, 2, 1, 12, 1, 64]) )
#past_key_values = torch.zeros( torch.Size([24, 12, 1, 64]) )
past_key_values = torch.zeros( torch.Size([24, 1, 12, 1, 64]) )
extended_attention_mask = copy.deepcopy(extended_attention_mask_init)

past_key_values = copy.deepcopy(past_key_values_init)
#past_key_values_flat = torch.stack(
#    [torch.cat([pk[0].unsqueeze(0), pk[1].unsqueeze(0)], dim=0) for pk in past_key_values], dim=0)
#past_key_values_flat = torch.cat(
#    [torch.cat(pk, dim=0) for pk in past_key_values], dim=0)
past_key_values_flat = torch.cat(
    [torch.stack(pk, dim=0) for pk in past_key_values], dim=0)
print(past_key_values_flat.shape)


with torch.no_grad():
    '''
    past_key_values = None
    output_id, past_key_values = wrapper(input_ids, attention_mask, past_key_values)
    print("First pass:")
    print("output_id:", output_id)
    print("input_ids:", input_ids.shape)
    print("attention_mask:", attention_mask.shape)
    print("past_key_values:", past_key_values.shape)
    '''

    #output_id, past_key_values_2 = wrapper(new_input_ids, extended_attention_mask, past_key_values_flat)
    logits, past_key_values_2 = wrapper(new_input_ids, extended_attention_mask, past_key_values_flat)
    output_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    print("Second pass:")
    print("output_id:", output_id)
    print("new_input_ids:", new_input_ids.shape)
    print("extended_attention_mask:", extended_attention_mask.shape)
    print("past_key_values:", past_key_values_flat.shape)
    print("past_key_values_2:", past_key_values_2.shape)


past_key_values_zero = torch.zeros( torch.Size([12, 2, 1, 12, 1, 64]) )
attention_mask_zero = torch.cat([attention_mask, torch.ones(1,1)], dim=-1)
extended_attention_mask = extended_attention_mask.to(torch.int8)
print(attention_mask_zero.shape)
# Export to ONNX
torch.onnx.export(
    wrapper,
    (new_input_ids, extended_attention_mask, past_key_values_flat),
    "onnx_model/gpt2_with_kv.onnx",
    input_names=["input_ids", "attention_mask", "past_key_values_input"],
    output_names=["output_id", "past_key_values_output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "new_sequence"},
        "attention_mask": {0: "batch_size", 1: "new_and_last_sequence"},
        "output_id": {0: "batch_size"},
        "past_key_values_input": {1: "batch_size", 3: "last_sequence"},
        "past_key_values_output": {1: "batch_size", 3: "new_and_last_sequence"}
    },
    opset_version=11
)

#"""
import onnx
import onnxruntime as ort

# Verify the exported ONNX model
onnx_model = onnx.load("../../../../modclub/rust-connect-py-ai-to-ic/python/onnx_model/gpt2_with_kv.onnx")
onnx.checker.check_model(onnx_model)

# Initialize ONNX Runtime session
ort_session = ort.InferenceSession("../../../../modclub/rust-connect-py-ai-to-ic/python/onnx_model/gpt2_with_kv.onnx")


#next_input = torch.cat([input_ids, new_input_ids], dim=1)
#past_key_values = torch.zeros( torch.Size([12, 2, 1, 12, 1, 64]) )
extended_attention_mask = copy.deepcopy(extended_attention_mask_init)
extended_attention_mask = extended_attention_mask.to(torch.int8)
past_key_values = copy.deepcopy(past_key_values_init)
#past_key_values_flat = torch.stack(
#    [torch.cat([pk[0].unsqueeze(0), pk[1].unsqueeze(0)], dim=0) for pk in past_key_values], dim=0)
#past_key_values_flat = torch.cat(
#    [torch.cat(pk, dim=0) for pk in past_key_values], dim=0)
past_key_values_flat = torch.cat(
    [torch.stack(pk, dim=0) for pk in past_key_values], dim=0)
# Prepare dummy inputs for ONNX
next_input_ort = new_input_ids.numpy()
extended_attention_mask_ort = extended_attention_mask.numpy()
past_key_values_ort = past_key_values_flat.numpy()

print(next_input_ort)
print(next_input_ort.shape)
print(extended_attention_mask_ort.shape)
print(past_key_values_ort.shape)

# Create input feed dictionary
onnx_inputs = {
    "input_ids": next_input_ort,
    "attention_mask": extended_attention_mask_ort,
    "past_key_values_input": past_key_values_ort
}

output_ids = []
#for i in range(3):
# Run inference
outputs = ort_session.run(None, onnx_inputs)

# Verify the outputs
output_id_ort = outputs[0]
past_key_values_ort = outputs[1]
print(f"Output ID: {output_id_ort}")
print(f"Past Key Values: {past_key_values_ort.shape}")


#########################################################

import torch

next_input = torch.cat([input_ids, new_input_ids], dim=1)
next_input_ort = next_input.numpy()

past_key_values_zero = torch.zeros( torch.Size([12, 2, 1, 12, 1, 64]) )


extended_attention_mask = copy.deepcopy(extended_attention_mask_init)
extended_attention_mask = extended_attention_mask.to(torch.int8)
extended_attention_mask_ort = extended_attention_mask.numpy()
past_key_values_ort = past_key_values_zero.numpy()

print(next_input_ort)
print(next_input_ort.shape)
print(extended_attention_mask_ort.shape)
print(past_key_values_ort.shape)

# Create input feed dictionary
onnx_inputs = {
    "input_ids": next_input_ort,
    "attention_mask": extended_attention_mask_ort,
    "past_key_values_input": past_key_values_ort
}



output_ids = []
for i in range(15):

    outputs = ort_session.run(None, onnx_inputs)
    # Verify the outputs
    output_id_ort = outputs[0]
    past_key_values_ort = outputs[1]
    print(f"Output ID: {output_id_ort}")
    print(f"Past Key Values: {past_key_values_ort.shape}")
    next_input = torch.tensor([[current_out]])
    extended_attention_mask = torch.cat([ extended_attention_mask, torch.ones((1,1), dtype=extended_attention_mask.dtype) ], dim=1) \
    extended_attention_mask_ort = extended_attention_mask.numpy()

    # Create input feed dictionary
    onnx_inputs = {
        "input_ids": next_input_ort,
        "attention_mask": extended_attention_mask_ort,
        "past_key_values_input": past_key_values_ort
    }
