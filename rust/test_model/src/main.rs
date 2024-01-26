

use tract_onnx::prelude::*;
use tract_onnx::FrameworkExtension;
use anyhow::Error; // Import the Error type from anyhow
//use tract_nnef::prelude::*; // Import necessary traits and types from tract_nnef
//use tract_nnef::nnef;
use std::sync::Arc; // For Arc
//use anyhow::Error; // Ensure you have the anyhow crate
//static STATE_ENCODED: &[u8] = include_bytes!("../model/demo_regression.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../model/simple_gpt2.onnx");
use std::path::Path; // To specify the directory path

//use std::fs;
use std::io::Result;
/*
fn load_onnx_model() -> Result<Vec<u8>> {
    // Path to your ONNX model
    let path = "model/demo_regression.onnx";

    // Read the file's contents into a byte vector
    let model_data = fs::read(path)?;

    Ok(model_data)
}
*/
/*
let path = "model/demo_regression.onnx";
// Read the file's contents into a byte vector
let model_data = fs::read(path)?;
let model_bytes = bytes::Bytes::from(model_data);

 */
//static STATE_ENCODED: &[u8] = include_bytes!("../model/demo_regression.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../model/simple_gpt2.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../model/gpt2_lm_head.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../model/ResNet-50.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../model/llama2_7b_0_embed.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../model/llama2_7b_1_layer.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../model/llama2_7b_1_layer.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../model/mini_llama2_7b_mean.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../model/mini_mistral_7b_mean.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../../../Experiments/ONNX_Model/mini_gpt2_mean.onnx");
//static STATE_ENCODED: &[u8] = include_bytes!("../../../Experiments/ONNX_Model/wte_gpt2_mean.onnx");

/*
static STATE_ENCODED: &[u8] = include_bytes!("../../../Experiments/ONNX_Model/gpt2_phase_1.onnx");
static STATE_ENCODED_2: &[u8] = include_bytes!("../../../Experiments/ONNX_Model/gpt2_phase_2.onnx");
static STATE_ENCODED_3: &[u8] = include_bytes!("../../../Experiments/ONNX_Model/gpt2_phase_3.onnx");
static STATE_ENCODED_4: &[u8] = include_bytes!("../../../Experiments/ONNX_Model/gpt2_phase_4_mean.onnx");
*/

static STATE_ENCODED: &[u8] = include_bytes!("../../../Experiments/ONNX_Model/gpt2_phase_1_pooled.onnx");
static STATE_ENCODED_2: &[u8] = include_bytes!("../../../Experiments/ONNX_Model/demo_regression.onnx");
static STATE_ENCODED_3: &[u8] = include_bytes!("../../../Experiments/ONNX_Model/demo_regression.onnx");
static STATE_ENCODED_4: &[u8] = include_bytes!("../../../Experiments/ONNX_Model/demo_regression.onnx");

use smallvec::SmallVec;
type NN_In_Out = smallvec::SmallVec<[tract_core::value::TValue; 4]>;
type Tensor = tract_data::tensor::Tensor;

use std::time::Instant;

fn sub_compute(input: NN_In_Out) -> NN_In_Out {
    let model_bytes_4 = bytes::Bytes::from_static(STATE_ENCODED_4);
    let bytes_model_4 = tract_onnx::onnx().model_for_bytes(model_bytes_4.clone()).expect("Failed to Build Model").into_optimized().expect("Failed to Optimize Model").into_runnable().expect("Failed to make Model Runnable");
    let output = bytes_model_4.run(input).expect("Failed to Run Model");
    //println!("results4 {:?}", result_4);
    output
}


fn main() {

   //let st = serialize_tensor();

   //let snn = serialize_nn_in_out();
   // chained()

    //let cr = chained();

    let sr = single_model();

}




fn single_model() -> TractResult<()> {

    /*
    // Directly assigning the result of load_onnx_model to model_data
    let model_data = load_onnx_model()?;
    // At this point, model_data contains the contents of the ONNX model file
    println!("Model loaded successfully. Size: {} bytes", model_data.len());
    let model_bytes = bytes::Bytes::from(model_data);
     */
    let model_bytes = bytes::Bytes::from_static(STATE_ENCODED);


    /*
    match load_onnx_model() {
        Ok(model_data) => {
            // Model is now loaded into `model_data`
            println!("Model loaded successfully. Size: {} bytes", model_data.len());
            // You can now use `model_data` as needed
        }
        Err(e) => {
            // Handle the error
            eprintln!("Failed to load model: {}", e);
        }
    }
    */


    /*
    let model = tract_onnx::onnx()
        // load the model
        //.model_for_path("model/simple_gpt2.onnx")?
        // demo for memory
        .model_for_path("model/demo_regression.onnx")?
        // optimize the model
        .into_optimized()? //when converting to nnef do not make runnable
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;
    */


    // Create an Nnef object
    //let nnef = nnef();
    /*
    let nnef = tract_nnef::nnef().with_onnx();
    let model_dir = Path::new("model");

    // Serialize the model to a buffer
    let mut buffer = vec![];
    nnef.write_to_tar(&model, &mut buffer)?;
    //nnef.write_to_dir(&model, &mut buffer)?;
    println!("Reload from NNEF");

    // Reload the model from the buffer
    let reloaded = nnef.model_for_read(&mut &*buffer)?
        //.into_optimized()?
        .into_runnable()?;
    */

    /*
    // Your token IDs can be of variable length
    //let token_ids: Vec<i64> = vec![1, 2, 3]; // This can be of any length
    let token_ids: Vec<i64> = vec![10814,   612,     0,  1680,   345, 11241,  1096,   428,   329,   502,   30];
    println!("Token IDs {:?}", token_ids);

    // Determine the shape dynamically
    let shape = (1, token_ids.len()); // Shape is now (1, length of token_ids)

    // Convert your Vec<i64> into a Tensor
    let input_tensor = tract_ndarray::Array::from_shape_vec(shape, token_ids)
        .map_err(|e| Error::msg(format!("Error creating ndarray: {}", e)))?
        .into_tensor();
    */

    //let model_bytes: Bytes = /* load your model bytes here */;
    //let bytes_model = tract_onnx::onnx().model_for_bytes(model_bytes)?.into_runnable()?;
    //println!("Loaded Model");

    ///* below is the main code for GPT2 and Demo
    /* for a demo version
    let token_ids: Vec<i64> = vec![3];
    let shape = (1, token_ids.len()); // Shape is now (1, length of token_ids)
    // Convert token_ids to floats if needed
    let token_ids_f32: Vec<f32> = token_ids.iter().map(|&x| x as f32).collect();

    // Adjust the shape to match the model's expected input
    let shape = (1, token_ids_f32.len());

    let input_tensor = tract_ndarray::Array::from_shape_vec(shape, token_ids_f32)
        .map_err(|e| Error::msg(format!("Error creating ndarray: {}", e)))?
        .into_tensor();
    // demo version end

    println!("Input Tensor {:?}", input_tensor);
    // Assuming input_tensor is already created
    let copied_tensor = input_tensor.clone();


    // Run the model on the input tensor
    //let result = model.run(tvec!(input_tensor.into()))?;
    //let result = reloaded.run(tvec!(input_tensor.into()))?;
    */

    // Process the result as needed
    // ...
    //println!("results {:?}", result);


    // token Ids input (works for GPT2 but not yet for Llama2-mini)
    //let token_ids: Vec<i64> = vec![1, 2, 3]; // This can be of any length
    //let token_ids: Vec<i64> = vec![10814,   612,     0,  1680,   345, 11241,  1096,   428,   329,   502,   30];
    let token_ids: Vec<i64> = vec![    1, 15043, 29892,   920,   526,   366, 29973];
    println!("Token IDs {:?}", token_ids);

    // Determine the shape dynamically
    let shape = (1, token_ids.len()); // Shape is now (1, length of token_ids)

    // Convert your Vec<i64> into a Tensor
    let input_tensor = tract_ndarray::Array::from_shape_vec(shape, token_ids)
        .map_err(|e| Error::msg(format!("Error creating ndarray: {}", e)))?
        .into_tensor();

    println!("Input Tensor Type");
    print_type_of(&input_tensor);



    //let model_bytes: Bytes = /* load your model bytes here */;
    let bytes_model = tract_onnx::onnx().model_for_bytes(model_bytes.clone())?;
    println!("Just Reading");
    print_type_of(&bytes_model);

    let bytes_model = tract_onnx::onnx().model_for_bytes(model_bytes.clone())?.into_optimized()?;
    println!("And Optimized");
    print_type_of(&bytes_model);



    let bytes_model = bytes_model.into_runnable()?;
    println!("And Runnable");
    print_type_of(&bytes_model);
    println!("Loaded Model");

    let start = Instant::now();

    //let tensor_prepped = tvec!(input_tensor.clone().into());
    //println!("Tensor Prepped Type");
    //print_type_of(&tensor_prepped);

    let result = bytes_model.run(tvec!(input_tensor.into()))?;
    //println!("results {:?}", result);

    println!("results type");
    print_type_of(&result);
    print_type_of(&result[0]);
    print_type_of(&tvec!(result[0].clone()));

    let duration = start.elapsed();

    // Print the time taken
    println!("Time taken: {:?}", duration);

    println!("results {:?}", result);
    Ok(())
    //Ok(())
}


/*
use serde::{Serialize, Deserialize};
use bincode;

// Assuming Tensor and TValue are serializable
fn serialize_tensor() -> Vec<u8> {

    let token_ids: Vec<i64> = vec![    1, 15043, 29892,   920,   526,   366, 29973];
    println!("Token IDs {:?}", token_ids);

    // Determine the shape dynamically
    let shape = (1, token_ids.len()); // Shape is now (1, length of token_ids)

    // Convert your Vec<i64> into a Tensor
    let input_tensor = tract_ndarray::Array::from_shape_vec(shape, token_ids)
        .expect("Failed Tensor from Vec")
        .into_tensor();

    println!("Input Tensor Type");
    print_type_of(&input_tensor);

    bincode::serialize(&input_tensor).expect("Failed to serialize tensor")



}
*/


/*
fn serialize_nn_in_out() -> Vec<u8> {

    let token_ids: Vec<i64> = vec![    1, 15043, 29892,   920,   526,   366, 29973];
    println!("Token IDs {:?}", token_ids);

    // Determine the shape dynamically
    let shape = (1, token_ids.len()); // Shape is now (1, length of token_ids)

    // Convert your Vec<i64> into a Tensor
    let input_tensor = tract_ndarray::Array::from_shape_vec(shape, token_ids)
        .expect("Failed Tensor from Vec")
        .into_tensor();

    let nn_input1 = tvec!(input_tensor.into());
    let nn_input2 = tvec!(input_tensor);
    println!("Input Tensor Type");
    print_type_of(&input_tensor);
    print_type_of(&nn_input1);
    print_type_of(&nn_input2);


    let model_bytes = bytes::Bytes::from_static(STATE_ENCODED);
    let bytes_model = tract_onnx::onnx().model_for_bytes(model_bytes.clone()).expect("Failed to Create").into_optimized().expect("Failed to Optimize").into_runnable().expect("Failed to Runnable");
    let result = bytes_model.run(tvec!(input_tensor.into())).expect("Failed to Run");
    //println!("results {:?}", result);

    println!("results type");
    print_type_of(&result);
    print_type_of(&result[0]);
    print_type_of(&tvec!(result[0].clone()));

    bincode::serialize(&result).expect("Failed to serialize NN_In_Out")
}
*/





fn chained() -> TractResult<()> {

    /*
    // Directly assigning the result of load_onnx_model to model_data
    let model_data = load_onnx_model()?;
    // At this point, model_data contains the contents of the ONNX model file
    println!("Model loaded successfully. Size: {} bytes", model_data.len());
    let model_bytes = bytes::Bytes::from(model_data);
     */
    let model_bytes = bytes::Bytes::from_static(STATE_ENCODED);


    /*
    match load_onnx_model() {
        Ok(model_data) => {
            // Model is now loaded into `model_data`
            println!("Model loaded successfully. Size: {} bytes", model_data.len());
            // You can now use `model_data` as needed
        }
        Err(e) => {
            // Handle the error
            eprintln!("Failed to load model: {}", e);
        }
    }
    */


    /*
    let model = tract_onnx::onnx()
        // load the model
        //.model_for_path("model/simple_gpt2.onnx")?
        // demo for memory
        .model_for_path("model/demo_regression.onnx")?
        // optimize the model
        .into_optimized()? //when converting to nnef do not make runnable
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;
    */


    // Create an Nnef object
    //let nnef = nnef();
    /*
    let nnef = tract_nnef::nnef().with_onnx();
    let model_dir = Path::new("model");

    // Serialize the model to a buffer
    let mut buffer = vec![];
    nnef.write_to_tar(&model, &mut buffer)?;
    //nnef.write_to_dir(&model, &mut buffer)?;
    println!("Reload from NNEF");

    // Reload the model from the buffer
    let reloaded = nnef.model_for_read(&mut &*buffer)?
        //.into_optimized()?
        .into_runnable()?;
    */

    /*
    // Your token IDs can be of variable length
    //let token_ids: Vec<i64> = vec![1, 2, 3]; // This can be of any length
    let token_ids: Vec<i64> = vec![10814,   612,     0,  1680,   345, 11241,  1096,   428,   329,   502,   30];
    println!("Token IDs {:?}", token_ids);

    // Determine the shape dynamically
    let shape = (1, token_ids.len()); // Shape is now (1, length of token_ids)

    // Convert your Vec<i64> into a Tensor
    let input_tensor = tract_ndarray::Array::from_shape_vec(shape, token_ids)
        .map_err(|e| Error::msg(format!("Error creating ndarray: {}", e)))?
        .into_tensor();
    */

    //let model_bytes: Bytes = /* load your model bytes here */;
    //let bytes_model = tract_onnx::onnx().model_for_bytes(model_bytes)?.into_runnable()?;
    //println!("Loaded Model");

    ///* below is the main code for GPT2 and Demo
    /* for a demo version
    let token_ids: Vec<i64> = vec![3];
    let shape = (1, token_ids.len()); // Shape is now (1, length of token_ids)
    // Convert token_ids to floats if needed
    let token_ids_f32: Vec<f32> = token_ids.iter().map(|&x| x as f32).collect();

    // Adjust the shape to match the model's expected input
    let shape = (1, token_ids_f32.len());

    let input_tensor = tract_ndarray::Array::from_shape_vec(shape, token_ids_f32)
        .map_err(|e| Error::msg(format!("Error creating ndarray: {}", e)))?
        .into_tensor();
    // demo version end

    println!("Input Tensor {:?}", input_tensor);
    // Assuming input_tensor is already created
    let copied_tensor = input_tensor.clone();


    // Run the model on the input tensor
    //let result = model.run(tvec!(input_tensor.into()))?;
    //let result = reloaded.run(tvec!(input_tensor.into()))?;
    */

    // Process the result as needed
    // ...
    //println!("results {:?}", result);


    // token Ids input (works for GPT2 but not yet for Llama2-mini)
    //let token_ids: Vec<i64> = vec![1, 2, 3]; // This can be of any length
    //let token_ids: Vec<i64> = vec![10814,   612,     0,  1680,   345, 11241,  1096,   428,   329,   502,   30];
    let token_ids: Vec<i64> = vec![    1, 15043, 29892,   920,   526,   366, 29973];
    println!("Token IDs {:?}", token_ids);

    // Determine the shape dynamically
    let shape = (1, token_ids.len()); // Shape is now (1, length of token_ids)

    // Convert your Vec<i64> into a Tensor
    let input_tensor = tract_ndarray::Array::from_shape_vec(shape, token_ids)
        .map_err(|e| Error::msg(format!("Error creating ndarray: {}", e)))?
        .into_tensor();

    println!("Input Tensor Type");
    print_type_of(&input_tensor);



    //let model_bytes: Bytes = /* load your model bytes here */;
    let bytes_model = tract_onnx::onnx().model_for_bytes(model_bytes.clone())?;
    println!("Just Reading");
    print_type_of(&bytes_model);

    let bytes_model = tract_onnx::onnx().model_for_bytes(model_bytes.clone())?.into_optimized()?;
    println!("And Optimized");
    print_type_of(&bytes_model);



    let bytes_model = bytes_model.into_runnable()?;
    println!("And Runnable");
    print_type_of(&bytes_model);
    println!("Loaded Model");

    let start = Instant::now();

    //let tensor_prepped = tvec!(input_tensor.clone().into());
    //println!("Tensor Prepped Type");
    //print_type_of(&tensor_prepped);

    let result = bytes_model.run(tvec!(input_tensor.into()))?;
    //println!("results {:?}", result);

    println!("results type");
    print_type_of(&result);
    print_type_of(&result[0]);
    print_type_of(&tvec!(result[0].clone()));

    let duration = start.elapsed();

    // Print the time taken
    println!("Time taken: {:?}", duration);

    /*
    // Assuming result is a SmallVec of TValues, and each TValue can be treated as a tensor
    for (index, tensor) in result.iter().enumerate() {
        // Here, `shape` method is used as an example; the actual method may vary
        let shape = tensor.shape(); // Replace with the actual method to get the shape
        println!("Tensor {}: Shape {:?}", index, shape);
    }
    */


    let model_bytes_2 = bytes::Bytes::from_static(STATE_ENCODED_2);
    let bytes_model_2 = tract_onnx::onnx().model_for_bytes(model_bytes_2.clone())?.into_optimized()?.into_runnable()?;
    //let result_2 = bytes_model_2.run(tvec!(result[0].clone()))?;
    //let result_2 = bytes_model_2.run(result.clone())?;   // this works
    let result_2 = bytes_model_2.run(result)?;
    println!("results2 {:?}", result_2);

    let model_bytes_3 = bytes::Bytes::from_static(STATE_ENCODED_3);
    let bytes_model_3 = tract_onnx::onnx().model_for_bytes(model_bytes_3.clone())?.into_optimized()?.into_runnable()?;
    let result_3 = bytes_model_3.run(tvec!(result_2[0].clone()))?;
    println!("results3 {:?}", result_3);

    /*
    let model_bytes_4 = bytes::Bytes::from_static(STATE_ENCODED_4);
    let bytes_model_4 = tract_onnx::onnx().model_for_bytes(model_bytes_4.clone())?.into_optimized()?.into_runnable()?;
    let result_4 = bytes_model_4.run(tvec!(result_3[0].clone()))?;
    println!("results4 {:?}", result_4);
    */
    let result_4 = sub_compute(result_3);
    println!("results4 {:?}", result_4);

    /*
    // Assuming the result contains only one tensor and it's a 1D tensor of f32
    if let Some(first_tensor) = result.first() {
        if let Ok(values) = first_tensor.to_array_view::<f32>() {
            println!("Values: {:?}", values.as_slice().unwrap());
        } else {
            println!("Error converting tensor to array view");
        }
    } else {
        println!("No tensors in the result");
    }
    */


    Ok(())
    //Ok(())
}





fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>());
}

