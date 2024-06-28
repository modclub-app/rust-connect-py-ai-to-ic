use std::cell::RefCell;
use prost::Message;
use tract_onnx::prelude::*;
use anyhow::anyhow;
use crate::upload_utils::call_model_bytes;
use candid::{CandidType, Deserialize};


type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

thread_local! {
    static MODEL_PIPELINE: RefCell<Option<ModelPipeline>> = RefCell::new(None);
}

struct ModelPipeline {
    models: Vec<Model>,
}

impl ModelPipeline {
    fn add_model(&mut self, model: Model) {
        self.models.push(model);
    }

    pub fn clear(&mut self) {
        self.models.clear();
    }

    pub fn new() -> Self {
        ModelPipeline {
            models: Vec::new(),
        }
    }

    pub fn run_model_at_index(&self, index: u8, input_tensor: Tensor) -> Result<(Vec<f32>, Vec<usize>), anyhow::Error> {
        let model = &self.models[index as usize];
        let result = model.run(tvec!(input_tensor.into()))?;

        if let Some(first_tensor) = result.first() {
            let shape = first_tensor.shape().to_vec(); // Get the shape of the tensor
            match first_tensor.to_array_view::<f32>() {
                Ok(values) => Ok((values.as_slice().unwrap_or(&[]).to_vec(), shape)),
                Err(_) => Err(anyhow!("Failed to convert tensor to array view")),
            }
        } else {
            Err(anyhow!("No Data in Result"))
        }
    }
}



#[derive(CandidType, Deserialize)]
enum TensorInput {
    F32(Vec<f32>),
    I64(Vec<i64>),
}


#[derive(CandidType, Deserialize)]
enum ModelInferenceResult {
    Ok(Vec<f32>),
    Err(String),
}

//////////////////////////////////////////////////////////////////////////////////




#[ic_cdk::update]
pub fn pipeline_init() {
    MODEL_PIPELINE.with(|pipeline_ref| {
        let mut pipeline = pipeline_ref.borrow_mut();
        *pipeline = Some(ModelPipeline::new());
    });
}

#[ic_cdk::update]
fn pipeline_clear() {
    MODEL_PIPELINE.with(|pipeline_ref| {
        if let Some(ref mut pipeline) = *pipeline_ref.borrow_mut() {
            pipeline.clear();
        }
    });
}

fn add_model_to_pipeline(model: Model) {
    MODEL_PIPELINE.with(|pipeline_ref| {
        let mut pipeline = pipeline_ref.borrow_mut();
        if let Some(model_pipeline) = &mut *pipeline {
            model_pipeline.add_model(model);
        }
    });
}

/// Constructs a runnable model from the serialized ONNX model in `RedactorNET`.
pub fn setup() -> TractResult<()> {
    let bytes = match call_model_bytes() {
        Ok(value) => value,
        Err(_) => bytes::Bytes::new(),
    };
    let proto: tract_onnx::pb::ModelProto = tract_onnx::pb::ModelProto::decode(bytes)?;
    let model = tract_onnx::onnx()
        .model_for_proto_model(&proto)?
        .into_optimized()?
        .into_runnable()?;
    add_model_to_pipeline(model);
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
pub fn create_tensor_and_run_model_pipeline(input: Vec<i64>) -> Result<Vec<f32>, String> {
    let mut input_shape = vec![1, input.len()];
    let mut input_tensor = TensorInput::I64(input);

    let num_models = MODEL_PIPELINE.with(|pipeline_ref| {
        pipeline_ref.borrow().as_ref().map_or(0, |pipeline| pipeline.models.len())
    });

    for index in 0..num_models as u8 {
        let call_result = model_sub_compute(index, input_tensor.clone(), input_shape.clone()).await;

        match call_result {
            ModelInferenceResult::Ok(new_output, new_input_shape) => {
                input_shape = new_input_shape;
                input_tensor = TensorInput::F32(new_output);
            }
            ModelInferenceResult::Err(e) => {
                return Err(format!("No Data in Result: {}", e));
            }
        }
    }

    if let TensorInput::F32(output) = input_tensor {
        Ok(output)
    } else {
        Err("Final tensor is not of type f32".to_string())
    }
}



#[ic_cdk::query]
fn model_sub_compute(index: u8, input: TensorInput, input_shape: Vec<usize>) -> ModelInferenceResult {
    let input_tensor = match input {
        TensorInput::F32(data) => match Array::from_shape_vec(input_shape, data) {
            Ok(array) => array.into_tensor(),
            Err(_) => return ModelInferenceResult::Err("Failed to create tensor from shape and values".to_string()),
        },
        TensorInput::I64(data) => match Array::from_shape_vec(input_shape, data) {
            Ok(array) => array.into_tensor(),
            Err(_) => return ModelInferenceResult::Err("Failed to create tensor from shape and values".to_string()),
        },
    };

    let call_result = MODEL_PIPELINE.with(|pipeline_ref| {
        let pipeline = pipeline_ref.borrow();
        let model = pipeline.as_ref().ok_or_else(|| "Model pipeline is not initialized".to_string())?;
        model.run_model_at_index(index, input_tensor)
    });

    match call_result {
        Ok((output, _)) => ModelInferenceResult::Ok(output),
        Err(e) => ModelInferenceResult::Err(format!("Model computation failed: {}", e)),
    }
}