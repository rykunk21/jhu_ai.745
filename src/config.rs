//! This file represents the config type and a load function that just uses serde to deserialize the config into my struct

use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub schema: SchemaConfig,
    pub model_path: String,
    pub training: TrainingConfig,
    pub defaults: Defaults,
}

#[derive(Debug, Deserialize)]
pub struct SchemaConfig {
    pub entities: Vec<String>,
    pub relations: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct TrainingConfig {
    pub script_path: String,
    pub model_type: String,
    pub negative_sample_size: i32,
    pub batch_size: i32,
    pub hidden_dim: i32,
    pub gamma: f64,
    pub alpha: f64,
    pub learning_rate: f64,
    pub max_steps: i32,
    pub regularization: f64,
    pub save_path: String,
}

#[derive(Debug, Deserialize)]
pub struct Defaults {
    pub input: String,
    pub out: String,
}

pub fn load(path: &Path) -> Config {
    let text = fs::read_to_string(path).unwrap();
    serde_json::from_str(&text).unwrap()
}
