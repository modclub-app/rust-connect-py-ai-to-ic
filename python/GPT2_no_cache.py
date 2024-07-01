from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import onnx
import onnxruntime as ort

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare inputs
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Perform the initial model run
outputs = model(input_ids, attention_mask=attention_mask)
logits = outputs.logits

print("Logits:", logits.shape)

# Prepare model for export by creating a wrapper
class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPT2Wrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

# Initialize the wrapper
wrapper = GPT2Wrapper(model)

# Dummy inputs for tracing
batch_size = 1
seq_length = 6
input_ids = torch.randint(0, 50257, (batch_size, seq_length))  # Example input_ids
attention_mask = torch.ones((batch_size, seq_length))          # Example attention mask

# Trace and export the model to ONNX
torch.onnx.export(
    wrapper,
    (input_ids, attention_mask),
    "onnx_model/gpt2_without_kv_caching.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"}
    },
    opset_version=11
)

# Verify the exported ONNX model
onnx_model = onnx.load("onnx_model/gpt2_without_kv_caching.onnx")
onnx.checker.check_model(onnx_model)

# Initialize ONNX Runtime session
ort_session = ort.InferenceSession("onnx_model/gpt2_without_kv_caching.onnx")

# Prepare dummy inputs for ONNX
input_ids_ort = input_ids.numpy()
attention_mask_ort = attention_mask.numpy()

# Run inference
onnx_inputs = {
    "input_ids": input_ids_ort,
    "attention_mask": attention_mask_ort,
}
outputs = ort_session.run(None, onnx_inputs)

# Verify the outputs
logits_ort = outputs[0]
print(f"Logits: {logits_ort.shape}")
