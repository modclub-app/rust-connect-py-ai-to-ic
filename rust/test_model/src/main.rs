use anyhow::{Error, Result}; // Import Error and Result type from anyhow
use tract_onnx::prelude::*;
use tract_onnx::FrameworkExtension;
use std::time::Instant;



static STATE_ENCODED: &[u8] = include_bytes!("../../../python/onnx_model/simplest_good_model.onnx");


fn main() -> Result<()> {
    single_model()?;
    Ok(())
}

fn single_model() -> Result<()> {
    let model_bytes = bytes::Bytes::from_static(STATE_ENCODED);
    let token_ids: Vec<i64> = vec![1, 15043, 29892, 920, 526, 366, 29973];
    let shape = (1, token_ids.len());
    let input_tensor = tract_ndarray::Array::from_shape_vec(shape, token_ids)
        .map_err(|e| Error::msg(format!("Error creating ndarray: {}", e)))?
        .into_tensor();

    let bytes_model = tract_onnx::onnx().model_for_bytes(model_bytes.clone())?.into_optimized()?.into_runnable()?;

    let start = Instant::now();
    let result = bytes_model.run(tvec!(input_tensor.into()))?;
    println!("results {:?}", result);

    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);

    Ok(())
}

