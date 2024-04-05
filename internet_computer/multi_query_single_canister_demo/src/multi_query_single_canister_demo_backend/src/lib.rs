
use std::cell::RefCell;
use tract_onnx::FrameworkExtension;
use tract_onnx::prelude::tvec;
use tract_onnx::prelude::tract_ndarray;
use tract_onnx::prelude::IntoTensor;
use tract_onnx::prelude::InferenceModelExt;



#[ic_cdk::query]
fn get_canister_id() -> String {
    let canister_name = ic_cdk::api::id();
    ic_cdk::println!("Created canister {}", canister_name);
    canister_name
}


type SimplePlanTypeRead = tract_core::model::graph::Graph<
    tract_hir::infer::fact::InferenceFact,
    std::boxed::Box<
        dyn tract_hir::infer::ops::InferenceOp
    >
>;

type SimplePlanTypeRun = tract_core::plan::SimplePlan<
    tract_core::model::fact::TypedFact,
    std::boxed::Box<dyn tract_core::ops::TypedOp>,
    tract_core::model::graph::Graph<
        tract_core::model::fact::TypedFact,
        std::boxed::Box<dyn tract_core::ops::TypedOp>
    >
>;

type Tensor = tract_data::tensor::Tensor;

thread_local! {
    static WASM_REF_CELL: RefCell<Vec<u8>> = RefCell::new(vec![]);
    static MODEL_READ_REF_CELL: RefCell<Option<SimplePlanTypeRead>> = RefCell::new(None);
    static MODEL_PIPELINE: RefCell<Option<ModelPipeline>> = RefCell::new(None);
    //static MODEL_PIPELINE: RefCell<Option<Vec<SimplePlanTypeRun>>> = RefCell::new(None);

}


/*
##################
Model Pipeline
###################
*/

struct ModelPipeline {
    models: Vec<SimplePlanTypeRun>,
}


impl ModelPipeline {
    fn add_model(&mut self, model: SimplePlanTypeRun) {
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

    pub fn run_model_at_index(&self, index: u8, input_tensor: Tensor) -> Result<(Vec<f32>, Vec<usize>), String> {

        let model = &self.models[index as usize];
        let result = model.run(tvec!(input_tensor.into()))
            .map_err(|e| e.to_string())?; // Handle the error properly

        if let Some(first_tensor) = result.first() {
            let shape = first_tensor.shape().to_vec(); // Get the shape of the tensor
            match first_tensor.to_array_view::<f32>() {
                Ok(values) => Ok((values.as_slice().unwrap_or(&[]).to_vec(), shape)),
                Err(_) => Err("Failed to convert tensor to array view".to_string()),
            }
        } else {
            Err("No Data in Result".to_string())
        }
    }

}

#[ic_cdk::update]
fn initialize_model_pipeline() {
    MODEL_PIPELINE.with(|pipeline_ref| {
        let mut pipeline = pipeline_ref.borrow_mut();
        *pipeline = Some(ModelPipeline::new()); // Assuming ModelPipeline has a new() method
    });
}

fn add_model_to_pipeline(model: SimplePlanTypeRun) {
    MODEL_PIPELINE.with(|pipeline_ref| {
        let mut pipeline = pipeline_ref.borrow_mut();
        if let Some(model_pipeline) = &mut *pipeline {
            model_pipeline.add_model(model);
        }
    });
}



/*
#############################
Uploading and Initializing Model
#############################
*/

#[ic_cdk::update]
pub fn upload_model_chunks(bytes: Vec<u8>) { // -> Result<(), String>
    WASM_REF_CELL.with(|wasm_ref_cell| {
        let mut wasm_ref_mut = wasm_ref_cell.borrow_mut();
        wasm_ref_mut.extend(bytes);
    });
}

#[ic_cdk::update]
fn model_bytes_to_plan() {
    let model_bytes = match call_model_bytes() {
        Ok(value) => value,
        Err(_) => bytes::Bytes::new(),  // Return empty bytes in case of error
    };
    let bytes_model = tract_onnx::onnx().model_for_bytes(model_bytes).expect("Failed to Read");
    MODEL_READ_REF_CELL.with(|model_ref_cell| {
        *model_ref_cell.borrow_mut() = Some(bytes_model);
    });
}

fn call_model_bytes() -> Result<bytes::Bytes, String> {
    WASM_REF_CELL.with(|wasm_ref_cell| {
        // Use `std::mem::take` to replace the contents of the cell with an empty vector
        // and return the original contents.
        let wasm_ref = std::mem::take(&mut *wasm_ref_cell.borrow_mut());

        // Convert the vector into a `Bytes` instance. This avoids cloning as the vector's
        // ownership is transferred to the `Bytes` instance.
        let model_bytes = bytes::Bytes::from(wasm_ref);

        Ok(model_bytes)
    })
}


#[ic_cdk::update]
fn plan_to_running_model() {

    MODEL_READ_REF_CELL.with(|read_cell| {
        // Take the model out of the RefCell, leaving None in its place
        if let Some(read_model) = read_cell.borrow_mut().take() {
            let run_model = read_model.into_optimized()
                .expect("Couldn't Optimize")
                .into_runnable()
                .expect("Couldn't Make Runnable");

            // now want to do this
            add_model_to_pipeline(run_model);


        }
    })
}



/*
#############################
Canister Queries
#############################
*/






#[ic_cdk::query(composite = true)]
async fn word_embeddings(input_text: String) -> Vec<f32> {

    // Trim, remove brackets, split by comma, and parse each number
    let numbers_result: Result<Vec<i64>, _> = input_text
        .trim()                        // Trim whitespace
        .trim_matches(|c| c == '[' || c == ']')  // Remove brackets
        .split(',')                    // Split by comma
        .map(|num_str| num_str.trim().parse::<i64>()) // Parse each number
        .collect();                    // Collect into a Result<Vec<i64>, E>

    // Handle the result
    let numbers = match numbers_result {
        Ok(vec) => vec,
        Err(e) => {
            // Handle the error
            //eprintln!("Failed to parse number: {}", e);
            ic_cdk::println!("Failed to parse number: {}", e);
            vec![]  // Return an empty vector or handle as needed
        },
    };

    let output:Vec<f32> = run_model_and_get_result_chain(numbers).await;
    output

}


async fn run_model_and_get_result_chain(token_ids: Vec<i64>) -> Vec<f32> {
    let input_shape = vec![1, token_ids.len()];
    //let shape: Vec<u64> = input_shape.into_iter().map(|dim| dim as u64).collect();
    //let (mut out, mut result_shape) = sub_nn_compute_i64(0, token_ids, input_shape);
    //let zero_nat8 = 0_u8;

    let (mut out, mut result_shape) = match ic_cdk::call(ic_cdk::api::id(), "sub_nn_compute_i64", (0_u8, token_ids, input_shape)).await {
        Ok(r) => {
            let (iner_out, iner_result_shape): (Vec<f32>, Vec<usize>) = r;
            (iner_out, iner_result_shape)
        },
        Err(e) => {
            ic_cdk::println!("Call to sub_nn_compute_i64 failed: {:?}", e);
            (vec![69.0,42.0], vec![1,1,2])
        }
    };

    let num_models = MODEL_PIPELINE.with(|pipeline_ref| {
        pipeline_ref.borrow().as_ref().map_or(0, |pipeline| pipeline.models.len())
    });

    for index in 1..num_models as u8 {
        // Temporarily move out and result_shape to call_result
        let call_result = ic_cdk::call(ic_cdk::api::id(), "sub_nn_compute_f32", (index, std::mem::take(&mut out), std::mem::take(&mut result_shape))).await;

        match call_result {
            Ok(r) => {
                let (iner_out, iner_result_shape): (Vec<f32>, Vec<usize>) = r;
                out = iner_out;
                result_shape = iner_result_shape;
            },
            Err(e) => {
                ic_cdk::println!("Call to sub_nn_compute_f32 failed: {:?}", e);
                break;
            }
        };
    }

    out
}



#[ic_cdk::query]
fn sub_nn_compute_i64(index: u8, input: Vec<i64>, input_shape: Vec<usize>) -> (Vec<f32>, Vec<usize>) {

    let input_tensor = match tract_ndarray::Array::from_shape_vec(input_shape, input) {
        Ok(array) => array.into_tensor(),
        Err(_) => Tensor::default(), // or handle error appropriately
    };

    run_model(index, input_tensor)

}

#[ic_cdk::query]
fn sub_nn_compute_f32(index: u8, input: Vec<f32>, input_shape: Vec<usize>) -> (Vec<f32>, Vec<usize>) {

    let input_tensor = match tract_ndarray::Array::from_shape_vec(input_shape, input) {
        Ok(array) => array.into_tensor(),
        Err(_) => Tensor::default(), // or handle error appropriately
    };

    run_model(index, input_tensor)
}





fn run_model(index: u8, input: Tensor) -> (Vec<f32>, Vec<usize>) {

    MODEL_PIPELINE.with(|pipeline_ref| {
        let pipeline = pipeline_ref.borrow();
        if let Some(model_pipeline) = &*pipeline {
            match model_pipeline.run_model_at_index(index, input) {
                Ok(result_tensor) => result_tensor, // directly use the result_tensor
                Err(_) => (vec![99.0, 100.0], vec![1, 2, 1]), // Return an error vector on failure to calculate
            }
        } else {
            // You need to return something if the model pipeline is not initialized
            (vec![6.0, 6.0, 6.0], vec![1, 1, 3]) // or handle this case as appropriate
        }
    })

}



