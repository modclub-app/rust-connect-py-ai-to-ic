#"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import copy
# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare initial inputs
input_text = "What is your favorite"
inputs = tokenizer(input_text, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

with torch.no_grad():
    # Perform the initial model run to get past_key_values
    outputs = model(input_ids, attention_mask=attention_mask)
logits = outputs.logits
print(logits.shape)
past_key_values = outputs.past_key_values


past_key_values_init = copy.deepcopy(past_key_values)

#print("Initial Logits:", logits.shape)
#print("Initial Past Key Values Shapes:", [pkv[0].shape for pkv in past_key_values])

# Prepare inputs for the second run
new_input_text = " sport?\n"
new_inputs = tokenizer(new_input_text, return_tensors='pt')
new_input_ids = new_inputs['input_ids']
print('Next IDs', new_input_ids)
# Extend the attention mask to account for past key values
extended_attention_mask = torch.cat(
    [attention_mask, torch.ones((attention_mask.size(0), new_input_ids.size(-1)), dtype=attention_mask.dtype)], dim=-1
)

extended_attention_mask_init = copy.deepcopy(extended_attention_mask)

##################################################################################################

#past_key_values = past_key_values_init
extended_attention_mask = copy.deepcopy(extended_attention_mask_init)


next_input = torch.cat([input_ids, new_input_ids], dim=1)
output_ids = []
# Run the model with past_key_values and extended attention mask
for i in range(15):
    with torch.no_grad():
        outputs = model(next_input, attention_mask=extended_attention_mask)
    current_out = torch.argmax(outputs.logits[:,-1,:]).item()
    output_ids.append(  current_out )
    #print('Output:', output_ids)
    next_input = torch.cat([next_input, torch.tensor([[current_out]])], dim=1)
    extended_attention_mask = torch.cat([ extended_attention_mask, torch.ones((1,1)) ], dim=1)
    #print('Ext Attn Mask Shape:', extended_attention_mask.shape)

#print(logits.shape)
#print(extended_attention_mask.shape)

print(output_ids)

response = torch.tensor(output_ids).long()

# Decode the encoded input IDs back to text
decoded_terms = tokenizer.decode(response)
print(f"No KV Decoded Terms: {decoded_terms}")

#########################################################################
#print(extended_attention_mask.shape)
#print(new_input_ids.shape)

next_input = copy.deepcopy(new_input_ids)
extended_attention_mask = copy.deepcopy(extended_attention_mask_init)
past_key_values = copy.deepcopy(past_key_values_init)

output_ids = []
# Run the model with past_key_values and extended attention mask
for i in range(15):
    with torch.no_grad():
        outputs = model(next_input, attention_mask=extended_attention_mask, past_key_values=past_key_values, use_cache=True)
    current_out = torch.argmax(outputs.logits[:,-1,:]).item()
    output_ids.append(  current_out )
    #print('Output:', output_ids)
    next_input = torch.tensor([[current_out]])
    past_key_values = outputs.past_key_values
    extended_attention_mask = torch.cat([ extended_attention_mask, torch.ones((1,1)) ], dim=1)
    #print('Ext Attn Mask Shape:', extended_attention_mask.shape)

#print(logits.shape)
#print(extended_attention_mask.shape)

print(output_ids)

response = torch.tensor(output_ids).long()

# Decode the encoded input IDs back to text
decoded_terms = tokenizer.decode(response)
print(f"KV Decoded Terms: {decoded_terms}")

########################################################################################


from transformers import GPT2LMHeadModel

class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPT2Wrapper, self).__init__()
        self.model = model
        self.num_layers = 12

    def forward(self, input_ids, attention_mask, past_key_values):

        # Reshape past_key_values from (num_layers * 2, batch_size, num_heads, seq_length, head_dim)
        past_key_values = tuple(
            (past_key_values[i], past_key_values[i + 1])
            for i in range(0, 2 * self.num_layers, 2)
        )

        outputs = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        #else:
        #    outputs = self.model(input_ids, attention_mask=attention_mask, past_key_values=None, use_cache=True)
        outputs_id = torch.argmax(outputs.logits[:,-1,:]).item()
        # Flatten past_key_values to a single tensor for ONNX export
        past_key_values_flat = torch.cat(
            [torch.stack(pk, dim=0) for pk in outputs.past_key_values], dim=0)
        #return logits, past_key_values_flat
        return outputs_id, past_key_values_flat

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


# Initialize the wrapper
wrapper = GPT2Wrapper(model)
wrapper.freeze_parameters()

next_input = copy.deepcopy(new_input_ids)
extended_attention_mask = copy.deepcopy(extended_attention_mask_init)
past_key_values = copy.deepcopy(past_key_values_init)
past_key_values_flat = torch.cat(
    [torch.stack(pk, dim=0) for pk in past_key_values], dim=0)
print(past_key_values_flat.shape)


output_ids = []
for i in range(15):
    with torch.no_grad():
        current_out, past_key_values_flat = wrapper(next_input, extended_attention_mask, past_key_values_flat)
    output_ids.append(  current_out )
    #print('Output:', output_ids)
    next_input = torch.tensor([[current_out]])
    past_key_values = past_key_values_flat
    extended_attention_mask = torch.cat([ extended_attention_mask, torch.ones((1,1)) ], dim=1)

response = torch.tensor(output_ids).long()

print(output_ids)

# Decode the encoded input IDs back to text
decoded_terms = tokenizer.decode(response)
print(f"KV Flattened Decoded Terms: {decoded_terms}")


######################################################################
# Show's Zero Init is Horrible Fail (If actually input)

next_input = torch.cat([input_ids, new_input_ids], dim=1)
extended_attention_mask = copy.deepcopy(extended_attention_mask_init)
#extended_attention_mask = torch.cat([ extended_attention_mask, torch.ones((1,1)) ], dim=1)
past_key_values_flat = torch.zeros( torch.Size([24, 1, 12, 1, 64]) )
'''
input_text = "<|endoftext|>"
inputs = tokenizer(input_text, return_tensors='pt')
eot = inputs['input_ids']
with torch.no_grad():
    outputs = model(eot, attention_mask=torch.ones((1,1), dtype=torch.int64), use_cache=True)

past_key_values_flat = torch.cat(
    [torch.stack(pk, dim=0) for pk in outputs.past_key_values], dim=0)
'''

#past_key_values_flat = torch.zeros( torch.Size([24, 12, 1, 64]) )
#print(past_key_values.shape)

output_ids = []

with torch.no_grad():
    current_out, past_key_values_flat = wrapper(next_input, torch.ones((1,next_input.shape[-1]+1), dtype=torch.int64), past_key_values_flat)
output_ids.append(current_out)
# print('Output:', output_ids)
print(past_key_values_flat.shape)
print(next_input.shape)

#next_input = torch.tensor([[current_out]])
#past_key_values = past_key_values_flat
extended_attention_mask = torch.ones((1, next_input.shape[1]+2), dtype=torch.int64)
print(extended_attention_mask.shape)
next_input = torch.tensor([[current_out]])

new_input_ids = copy.deepcopy(next_input)
for i in range(14):
    with torch.no_grad():
        current_out, past_key_values_flat = wrapper(next_input, extended_attention_mask, past_key_values_flat)
    output_ids.append(  current_out )
    #print('Output:', output_ids)
    next_input = torch.tensor([[current_out]])
    extended_attention_mask = torch.cat([ extended_attention_mask, torch.ones((1,1), dtype=extended_attention_mask.dtype) ], dim=1)

response = torch.tensor(output_ids).long()

print(output_ids)

# Decode the encoded input IDs back to text
decoded_terms = tokenizer.decode(response, skip_special_tokens=False)
print(f"KV Flattened Decoded Zero-Init Terms: {decoded_terms}")

'''
inputs = torch.cat([input_ids, new_input_ids], dim=1)
print('Inputs:', inputs)
print('Outputs:', output_ids)
# Convert the tensor to a list
inputs_list = inputs.tolist()
print('Inputs as list:', inputs_list)

# Check the format of inputs_list
if isinstance(inputs_list[0], list):
    # Flatten the list if it's nested
    inputs_list = [item for sublist in inputs_list for item in sublist]

print('Flattened Inputs as list:', inputs_list)
'''
# Now decode the terms
decoded_terms = tokenizer.decode(torch.tensor([2061,  318,  534, 4004, 6332, 198]).long(), skip_special_tokens=False)
print(f"Input Terms: {decoded_terms}")
# Inputs: tensor([[2061,  318,  534, 4004, 6332, 30, 198]])
# WITH ZERO  Outputs: [198, 2061, 318, 534, 4004, 6332, 30, 198, 198, 2061, 318, 534, 4004, 6332, 30]
# Standard : [198, 40, 1842, 284, 2342, 4346, 13, 314, 1842, 284, 2342, 9669, 13, 314, 1842]
