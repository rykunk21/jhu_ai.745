//! clap cli definition. I might want to use some of these command line options in the future

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
pub struct Cli {
    #[arg(long, default_value = "config.json")]
    pub config: PathBuf,

    #[arg(long)]
    pub input: Option<PathBuf>,

    #[arg(long)]
    pub out: Option<PathBuf>,

    #[arg(long)]
    pub train: bool,

    #[arg(long)]
    pub baseline: Option<String>,

    #[arg(long)]
    pub query: Option<String>,

    #[arg(long)]
    pub model_dir: Option<PathBuf>,

    #[arg(long)]
    pub data_dir: Option<PathBuf>,

    #[arg(long, default_value_t = 1.0)]
    pub alpha: f64,

    #[arg(long, default_value_t = 10)]
    pub top: usize,

    #[arg(long, default_value_t = 10)]
    pub baseline_top: usize,
}
