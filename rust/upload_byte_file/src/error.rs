

pub fn error_to_string(e: &dyn std::error::Error) -> String {
    format!("Upload Error: {}", e.to_string())
}


pub fn create_error_string(message: &str) -> String {
    format!("Upload Error: {message}")
}


