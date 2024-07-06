
use std::cell::RefCell;


thread_local! {
    //static WASM_REF_CELL: RefCell<Vec<u8>> = RefCell::new(vec![]);
    //static MODEL_READ_REF_CELL: RefCell<Option<SimplePlanTypeRead>> = RefCell::new(None);
    static MODEL_PIPELINE: RefCell<Option<ModelPipeline>> = RefCell::new(None);

}


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

#[ic_cdk::update]
fn clear_model_pipeline() {
    MODEL_PIPELINE.with(|pipeline_ref| {
        if let Some(ref mut pipeline) = *pipeline_ref.borrow_mut() {
            pipeline.clear();
        }
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
