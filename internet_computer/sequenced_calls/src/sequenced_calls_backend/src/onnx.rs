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



#[derive(CandidType, Deserialize, Clone, Debug)]
pub enum TensorInput {
    F32(Vec<f32>),
    I64(Vec<i64>),
}


#[derive(CandidType, Deserialize)]
pub enum ModelInferenceResult {
    Ok(Vec<f32>),
    Err(String),
}


//#[derive(CandidType, Deserialize)]
//pub enum ModelInferenceWithShapeResult {
//    Ok(Vec<f32>, Vec<usize>),
//    Err(String),
//}

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


pub async fn create_tensor_and_run_model_pipeline(input: Vec<i64>) -> ModelInferenceResult {
    let mut input_shape = vec![1, input.len()];
    let mut input_tensor = TensorInput::I64(input);

    let num_models = MODEL_PIPELINE.with(|pipeline_ref| {
        pipeline_ref.borrow().as_ref().map_or(0, |pipeline| pipeline.models.len())
    });

    for index in 0..num_models as u8 {
        ic_cdk::println!("Calling model_sub_compute with index: {}, input_tensor: {:?}, input_shape: {:?}", index, input_tensor, input_shape);

        let call_result: Result<(Vec<f32>, Vec<usize>), _> = ic_cdk::call(ic_cdk::api::id(), "model_sub_compute", (index, input_tensor.clone(), input_shape.clone())).await;

        match call_result {
            Ok((new_output, new_input_shape)) => {
                ic_cdk::println!("Call succeeded. new_output: {:?}, new_input_shape: {:?}", new_output, new_input_shape);
                input_shape = new_input_shape;
                input_tensor = TensorInput::F32(new_output);
            }
            Err(e) => {
                ic_cdk::println!("Call to model_sub_compute failed: {:?}", e);
                return ModelInferenceResult::Err("model_sub_compute failed".to_string());
            }
        }
    }

    if let TensorInput::F32(output) = input_tensor {
        ModelInferenceResult::Ok(output)
    } else {
        ModelInferenceResult::Err("Final tensor is not of type f32".to_string())
    }
}

#[ic_cdk::query]
fn model_sub_compute(index: u8, input: TensorInput, input_shape: Vec<usize>) -> (Vec<f32>, Vec<usize>) {
    ic_cdk::println!("model_sub_compute called with index: {}, input: {:?}, input_shape: {:?}", index, input, input_shape);

    let input_tensor = match input {
        TensorInput::F32(data) => {
            ic_cdk::println!("Processing F32 input");
            tract_ndarray::Array::from_shape_vec(input_shape.clone(), data.clone()).unwrap_or_default().into_tensor()
        }
        TensorInput::I64(data) => {
            ic_cdk::println!("Processing I64 input");
            tract_ndarray::Array::from_shape_vec(input_shape.clone(), data.clone()).unwrap_or_default().into_tensor()
        }
    };

    let call_result = MODEL_PIPELINE.with(|pipeline_ref| {
        pipeline_ref.borrow().as_ref().map_or_else(
            || {
                ic_cdk::println!("Model pipeline is not initialized");
                (vec![], vec![])
            },
            |model_pipeline| {
                ic_cdk::println!("Running model at index: {}", index);
                model_pipeline.run_model_at_index(index, input_tensor).unwrap_or_else(|e| {
                    ic_cdk::println!("Model computation failed: {:?}", e);
                    (vec![], vec![])
                })
            },
        )
    });

    ic_cdk::println!("model_sub_compute result: {:?}", call_result);
    call_result
}

/*
#[ic_cdk::query]
fn model_sub_compute(index: u8, input: TensorInput, input_shape: Vec<usize>) -> Result<(Vec<f32>, Vec<usize>), String> {
    let input_tensor = match input {
        TensorInput::F32(data) => match tract_ndarray::Array::from_shape_vec(input_shape.clone(), data.clone()) {
            Ok(array) => array.into_tensor(),
            Err(_) => return Err("Failed to create tensor from shape and values".to_string()),
        },
        TensorInput::I64(data) => match tract_ndarray::Array::from_shape_vec(input_shape.clone(), data.clone()) {
            Ok(array) => array.into_tensor(),
            Err(_) => return Err("Failed to create tensor from shape and values".to_string()),
        },
    };

    let call_result = MODEL_PIPELINE.with(|pipeline_ref| {
        pipeline_ref
            .borrow()
            .as_ref()
            .ok_or_else(|| "Model pipeline is not initialized".to_string())
            .and_then(|model_pipeline| {
                model_pipeline
                    .run_model_at_index(index, input_tensor)
                    .map_err(|e| format!("Model computation failed: {}", e))
            })
    });

    match call_result {
        Ok((output, output_shape)) => Ok((output, output_shape)),
        Err(e) => Err(e),
    }
}
*/

