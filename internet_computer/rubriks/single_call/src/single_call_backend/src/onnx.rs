use std::cell::RefCell;

use prost::Message;
use tract_onnx::prelude::*;
use anyhow::anyhow;
use crate::storage;
use crate::MODEL_FILE;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

thread_local! {
    static MODEL: RefCell<Option<Model>> = RefCell::new(None);
}

/// Constructs a runnable model from the serialized ONNX model.
pub fn setup() -> TractResult<()> {
    // Read the model bytes from the file.
    let bytes = storage::bytes(MODEL_FILE);

    // Decode the model proto.
    let proto = tract_onnx::pb::ModelProto::decode(bytes)
        .map_err(|e| anyhow!("Failed to decode model proto: {}", e))?;

    // Build the runnable model.
    let model = tract_onnx::onnx()
        .model_for_proto_model(&proto)?
        .into_optimized()?
        .into_runnable()?;

    // Store the model in the thread-local storage.
    MODEL.with(|m| {
        *m.borrow_mut() = Some(model);
    });

    Ok(())
}

#[ic_cdk::update]
fn setup_model() -> Result<(), String> {
    match setup() {
        Ok(_) => Ok(()),
        Err(err) => Err(format!("Failed to setup model: {}", err)),
    }
}

pub fn create_tensor_and_run_model(token_ids: Vec<i64>) -> Result<Vec<f32>, anyhow::Error> {
    MODEL.with_borrow(|model| {
        let model = model.as_ref().unwrap();

        let input_shape = vec![1, token_ids.len()];

        let tensor = match tract_ndarray::Array::from_shape_vec(input_shape, token_ids) {
            Ok(array) => array.into_tensor(),
            Err(_) => return Err(anyhow!("Failed to create tensor from shape and values")),
        };

        let result = model.run(tvec!(Tensor::from(tensor).into()))?;

        let scores: Vec<f32> = result[0]
            .to_array_view::<f32>()?
            .iter()
            .cloned()
            .collect();

        Ok(scores)
    })
}
