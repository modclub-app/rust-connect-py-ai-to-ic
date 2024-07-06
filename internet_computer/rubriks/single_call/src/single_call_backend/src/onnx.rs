use std::cell::RefCell;

use prost::Message;
use tract_onnx::prelude::*;
use anyhow::anyhow;
use crate::upload_utils::call_model_bytes;


type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

thread_local! {
    static MODEL: RefCell<Option<Model>> = RefCell::new(None);
}

//const REDACTOR_NET: &'static [u8] = include_bytes!("../assets/simplest_best_dynamic_model.onnx");
//const REDACTOR_NET: &'static [u8] = include_bytes!("../assets/mobilenetv2-7.onnx");

/// Constructs a runnable model from the serialized ONNX model in `RedactorNET`.
pub fn setup() -> TractResult<()> {
    //let bytes = bytes::Bytes::from_static(REDACTOR_NET);
    let bytes = match call_model_bytes() {
        Ok(value) => value,
        Err(_) => bytes::Bytes::new(),  // Return empty bytes in case of error
    };
    let proto: tract_onnx::pb::ModelProto = tract_onnx::pb::ModelProto::decode(bytes)?;
    let model = tract_onnx::onnx()
        .model_for_proto_model(&proto)?
        .into_optimized()?
        .into_runnable()?;
    MODEL.with_borrow_mut(|m| {
        *m = Some(model);
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
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs the model on the given image and returns top three labels.
//pub fn classify(image: Vec<u8>) -> Result<Vec<Classification>, anyhow::Error> {
pub fn create_tensor_and_run_model(token_ids: Vec<i64>) -> Result<Vec<f32>, anyhow::Error> {

    MODEL.with_borrow(|model| {
        let model = model.as_ref().unwrap();

        let input_shape = vec![1, token_ids.len()];
        //let input_shape = vec![token_ids.len()];

        let tensor = match tract_ndarray::Array::from_shape_vec(input_shape, token_ids) {
            Ok(array) => array.into_tensor(),
            Err(_) => return Err(anyhow!("Failed to create tensor from shape and values")),
        };

        //let tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        //    (image[(x as u32, y as u32)][c] as f32 / 255.0 - MEAN[c]) / STD[c]
        //});

        let result = model.run(tvec!(Tensor::from(tensor).into()))?;

        let scores: Vec<f32> = result[0]
            .to_array_view::<f32>()?
            .iter()
            .cloned()
            .collect();

        Ok(scores)
    })
}



