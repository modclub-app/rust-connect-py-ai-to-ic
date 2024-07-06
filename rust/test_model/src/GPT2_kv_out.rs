use tract_onnx::prelude::*;
use tract_ndarray::ArrayViewD;
use crate::tract_ndarray::Axis;

fn main() -> TractResult<()> {
    // Load the ONNX model
    let model = tract_onnx::onnx()
        .model_for_path("../../python/onnx_model/gpt2_with_kv_out.onnx")?
        .into_optimized()?
        .into_runnable()?;

    // Initialize input tokens and attention mask
    let mut input_ids: Vec<i64> = vec![122, 3064]; // Use appropriate initial token
    let mut attention_mask: Vec<f32> = vec![1.0, 1.0];

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
        let attention_mask_tensor = create_tensor_f32(&attention_mask)?;

        println!("Input IDs Tensor: {:?}", input_ids_tensor);
        println!("Attention Mask Tensor: {:?}", attention_mask_tensor);

        let inputs = tvec!(input_ids_tensor.into(), attention_mask_tensor.into());

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
        let past_key_values = outputs[1].to_array_view::<f32>()?;

        // Get the next token
        let next_token = argmax(logits)?;

        // Print the next token for debugging
        println!("Next token: {}", next_token);

        // Print the shape of past_key_values for debugging
        println!("Past Key Values Shape: {:?}", past_key_values.shape());

        // Append the next token and update the attention mask
        input_ids.push(next_token);
        attention_mask.push(1.0);
    }

    println!("Final input_ids: {:?}", input_ids);
    println!("Final attention_mask: {:?}", attention_mask);

    Ok(())
}

fn create_tensor_i64(data: &[i64]) -> TractResult<Tensor> {
    let shape = [1, data.len()];
    let array = tract_ndarray::Array::from_shape_vec(shape, data.to_vec())
        .map_err(|_| anyhow::anyhow!("Failed to create tensor from shape and values"))?;
    Ok(array.into_tensor())
}

fn create_tensor_f32(data: &[f32]) -> TractResult<Tensor> {
    let shape = [1, data.len()];
    let array = tract_ndarray::Array::from_shape_vec(shape, data.to_vec())
        .map_err(|_| anyhow::anyhow!("Failed to create tensor from shape and values"))?;
    Ok(array.into_tensor())
}

fn argmax(logits: ArrayViewD<f32>) -> TractResult<i64> {
    let last_token_logits = logits.index_axis(Axis(1), logits.shape()[1] - 1);
    Ok(last_token_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64)
}
