use tract_onnx::prelude::*;
use tract_ndarray::{Array, IxDyn, Axis};
use tract_ndarray::ArrayViewD;

fn main() -> TractResult<()> {
    // Load the ONNX model
    let model = tract_onnx::onnx()
        .model_for_path("../../python/onnx_model/gpt2_with_kv_in.onnx")?
        .into_optimized()?
        .into_runnable()?;

    // Initialize input tokens and attention mask
    let mut input_ids: Vec<i64> = vec![122, 3064]; // Use appropriate initial token
    let mut attention_mask: Vec<i64> = vec![1, 1];
    let mut past_key_values = create_empty_past_key_values(24, 1, 12, 1, 64)?;

    // Loop for text generation
    for j in 0..3 { // Example: 3 iterations
        println!(
            "Iteration: {}, Input IDs Length: {}, Attention Mask Length: {}",
            j,
            input_ids.len(),
            attention_mask.len()
        );

        // Convert input_ids and attention_mask to tensors
        let input_ids_tensor = create_tensor_i64(&input_ids)?;
        let attention_mask_tensor = create_tensor_i64(&attention_mask)?;

        // Print tensor details
        println!("Input IDs Tensor: {:?}", input_ids_tensor);
        println!("Input IDs Tensor Shape: {:?}", input_ids_tensor.shape());
        println!("Input IDs Tensor DType: {:?}", input_ids_tensor.datum_type());

        println!("Attention Mask Tensor: {:?}", attention_mask_tensor);
        println!("Attention Mask Tensor Shape: {:?}", attention_mask_tensor.shape());
        println!("Attention Mask Tensor DType: {:?}", attention_mask_tensor.datum_type());

        println!("Past Key Values Tensor: {:?}", past_key_values);
        println!("Past Key Values Tensor Shape: {:?}", past_key_values.shape());
        println!("Past Key Values Tensor DType: {:?}", past_key_values.datum_type());

        let inputs: TVec<TValue> = tvec!(input_ids_tensor.into(), attention_mask_tensor.into(), past_key_values.clone().into());

        // Debugging shapes
        for (i, input) in inputs.iter().enumerate() {
            println!("Input {}: {:?}", i, input.shape());
            println!("Input {} DType: {:?}", i, input.datum_type());
        }

        // Run the inference
        let outputs = match model.run(inputs) {
            Ok(o) => o,
            Err(e) => {
                println!("Model run failed: {:?}", e);
                return Err(e);
            }
        };

        // Extract logits and past_key_values
        let logits = outputs[0].to_array_view::<f32>()?;
        past_key_values = outputs[1].clone().into_tensor();

        // Get the next token
        let next_token = argmax(logits)?;

        // Print the next token for debugging
        println!("Next token: {}", next_token);

        // Print the shape of past_key_values for debugging
        println!("Past Key Values Shape: {:?}", past_key_values.shape());

        // Append the next token and update the attention mask
        input_ids.push(next_token);
        attention_mask.push(1);
    }

    println!("Final input_ids: {:?}", input_ids);
    println!("Final attention_mask: {:?}", attention_mask);

    Ok(())
}

fn create_tensor_i64(data: &[i64]) -> TractResult<Tensor> {
    let shape = [1, data.len()];
    let array = Array::from_shape_vec(IxDyn(&shape), data.to_vec())
        .map_err(|_| anyhow::anyhow!("Failed to create tensor from shape and values"))?;
    Ok(array.into_tensor())
}

fn create_empty_past_key_values(num_layers: usize, batch_size: usize, num_heads: usize, seq_length: usize, head_dim: usize) -> TractResult<Tensor> {
    let shape = [num_layers, batch_size, num_heads, seq_length, head_dim];
    let array = Array::from_shape_vec(IxDyn(&shape), vec![0.0_f32; num_layers * batch_size * num_heads * seq_length * head_dim])
        .map_err(|_| anyhow::anyhow!("Failed to create tensor from shape and values"))?;
    Ok(array.into_tensor())
}

fn argmax(logits: ArrayViewD<f32>) -> TractResult<i64> {
    let last_token_logits = logits.index_axis(Axis(1), logits.shape()[1] - 1);
    Ok(last_token_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
}
