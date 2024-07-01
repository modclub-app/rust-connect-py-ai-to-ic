import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

class GPT2_Embedding_Phase(torch.nn.Module):
    def __init__(self, original_model):
        super(GPT2_Embedding_Phase, self).__init__()
        self.wte = original_model.transformer.wte
        self.wpe = original_model.transformer.wpe

    def forward(self, input_ids):
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
        inputs_embeds = self.wte(input_ids) + self.wpe(position_ids)
        return inputs_embeds

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

class GPT2_Phase(torch.nn.Module):
    def __init__(self, original_model, layer):
        super(GPT2_Phase, self).__init__()
        # Handle a single transformer layer
        self.layer = original_model.transformer.h[layer]

    def forward(self, hidden_states):
        outputs = self.layer(hidden_states)
        hidden_states = outputs[0]
        return hidden_states

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

class GPT2_Output_Phase(torch.nn.Module):
    def __init__(self, original_model):
        super(GPT2_Output_Phase, self).__init__()
        self.lm_head = original_model.lm_head

    def forward(self, hidden_states):
        logits = self.lm_head(hidden_states)
        return logits

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

def split_and_export_model(original_model, model_name_prefix):
    input_ids = torch.tensor([[464, 1893]], dtype=torch.long)  # Example input

    # Export the embedding phase
    embedding_model = GPT2_Embedding_Phase(original_model)
    embedding_model.freeze_parameters()
    torch.onnx.export(embedding_model,
                      input_ids,
                      f"onnx_model/{model_name_prefix}_embedding.onnx",
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input_ids'],
                      output_names=['embedding_output'],
                      dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                    'embedding_output': {0: 'batch_size', 1: 'sequence_length'}})
    print(f"Exported: ONNX_Model/{model_name_prefix}_embedding.onnx")

    # Loop through and export each transformer layer
    hidden_states = embedding_model(input_ids)
    for layer in range(len(original_model.transformer.h)):
        layer_model = GPT2_Phase(original_model, layer)
        layer_model.freeze_parameters()

        # Export to ONNX
        model_path = f"onnx_model/{model_name_prefix}_layer_{layer}.onnx"
        torch.onnx.export(layer_model,
                          hidden_states,
                          model_path,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['hidden_states'],
                          output_names=['output'],
                          dynamic_axes={'hidden_states': {0: 'batch_size', 1: 'sequence_length'},
                                        'output': {0: 'batch_size', 1: 'sequence_length'}})
        print(f"Exported: {model_path}")

        # Update hidden states for the next layer
        hidden_states = layer_model(hidden_states)

    # Export the output phase
    output_model = GPT2_Output_Phase(original_model)
    output_model.freeze_parameters()
    torch.onnx.export(output_model,
                      hidden_states,
                      f"onnx_model/{model_name_prefix}_output.onnx",
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['hidden_states'],
                      output_names=['logits'],
                      dynamic_axes={'hidden_states': {0: 'batch_size', 1: 'sequence_length'},
                                    'logits': {0: 'batch_size', 1: 'sequence_length', 2: 'vocab_size'}})
    print(f"Exported: ONNX_Model/{model_name_prefix}_output.onnx")

# Example usage
split_and_export_model(model, model_name_prefix='gpt2')
