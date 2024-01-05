import torch
from transformers import GPT2Model

# Load the pre-trained GPT-2 model
model = GPT2Model.from_pretrained('gpt2')

# Keep only the first 6 blocks in the 'h' layer
model.h = torch.nn.ModuleList(model.h[:3])

class GPT2_Phase1(torch.nn.Module):
    def __init__(self, original_model):
        super(GPT2_Phase1, self).__init__()
        self.wte = original_model.wte
        self.wpe = original_model.wpe
        self.drop = original_model.drop
        self.h = original_model.h

    def forward(self, input_ids):
        # Handling position IDs
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Embedding layers
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Transformer blocks
        for block in self.h:
            outputs = block(hidden_states)
            hidden_states = outputs[0]

        # Mean of the last hidden state
        return hidden_states

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False



# Initialize the simplified model
phase_1_model = GPT2_Phase1(model)
phase_1_model.freeze_parameters()
# Example usage
input_ids = torch.tensor([[464, 1893]], dtype=torch.long)  # Example input

# Process through the simplified model
phase_1_output = phase_1_model(input_ids)

print(phase_1_output)



############################################################################################


# Load the pre-trained GPT-2 model
model = GPT2Model.from_pretrained('gpt2')



# Define a class for the second half of the GPT-2 model
class GPT2_Middle_Phase(torch.nn.Module):
    def __init__(self, original_model, start, end):
        super(GPT2_Middle_Phase, self).__init__()
        # Includes the second half of the transformer blocks and final layer norm
        self.h = original_model.h[start:end]  # Transformer blocks from h[6] to h[11]

    def forward(self, hidden_states):
        # Process through remaining transformer blocks
        for block in self.h:
            outputs = block(hidden_states)
            hidden_states = outputs[0]

        return hidden_states

    def freeze_parameters(self):
       for param in self.parameters():
            param.requires_grad = False


# Initialize the simplified model
GPT2_Phase2_model = GPT2_Middle_Phase(model, 3, 6)
GPT2_Phase2_model.freeze_parameters()
# Process through the simplified model
phase_2_output = GPT2_Phase2_model(phase_1_output)

GPT2_Phase3_model = GPT2_Middle_Phase(model, 6, 9)
GPT2_Phase3_model.freeze_parameters()
# Process through the simplified model
phase_3_output = GPT2_Phase3_model(phase_2_output)




######################

# Define a class for the second half of the GPT-2 model
class GPT2_End_Phase(torch.nn.Module):
    def __init__(self, original_model):
        super(GPT2_End_Phase, self).__init__()
        # Includes the second half of the transformer blocks and final layer norm
        self.h = original_model.h[9:]  # Transformer blocks from h[6] to h[11]
        self.ln_f = original_model.ln_f  # Final layer normalization

    def forward(self, hidden_states):
        # Process through remaining transformer blocks
        for block in self.h:
            outputs = block(hidden_states)
            hidden_states = outputs[0]

        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)

        return hidden_states

    def freeze_parameters(self):
       for param in self.parameters():
            param.requires_grad = False


# Initialize the simplified model
GPT2_End_Phase_model = GPT2_End_Phase(model)
GPT2_End_Phase_model.freeze_parameters()
# Process through the simplified model
output = GPT2_End_Phase_model(phase_3_output)

print(output)

test_output = model(input_ids)
print( test_output.last_hidden_state )

############################################################


# Export to ONNX
torch.onnx.export(phase_1_model,
                  input_ids,
                  "ONNX_Model/gpt2_phase_1.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input_ids'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size'}})



# Export to ONNX
torch.onnx.export(GPT2_Phase2_model,
                  phase_1_output,
                  "ONNX_Model/gpt2_phase_2.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input_ids'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size'}})

torch.onnx.export(GPT2_Phase3_model,
                  phase_2_output,
                  "ONNX_Model/gpt2_phase_3.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input_ids'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size'}})

torch.onnx.export(GPT2_End_Phase_model,
                  phase_3_output,
                  "ONNX_Model/gpt2_phase_4_mean.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input_ids'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size'}})




