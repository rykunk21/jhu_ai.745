mod baseline;
mod cli;
mod config;
mod query;
mod train;
mod types;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

fn main() -> Result<()> {
    // parse the cli and config
    let cli = cli::Cli::parse();
    let cfg = config::load(&cli.config);

    // If i pass a baseline query, run that and exit early
    if let Some(b) = &cli.baseline {
        baseline::run(b, cli.baseline_top)?;
        return Ok(());
    }

    // If i pass a query, run that and exit early
    if let Some(q) = &cli.query {
        let dir = cli
            .model_dir
            .unwrap_or_else(|| PathBuf::from(&cfg.training.save_path));
        let data = cli
            .data_dir
            .unwrap_or_else(|| PathBuf::from(&cfg.defaults.out));
        query::run(q, &dir, &data, cli.alpha, cli.top)?;
        return Ok(());
    }

    // run training by default
    let input = cli
        .input
        .unwrap_or_else(|| PathBuf::from(&cfg.defaults.input));
    let out = cli.out.unwrap_or_else(|| PathBuf::from(&cfg.defaults.out));

    // Extract with GLiNER
    train::run(&input, &out, &cfg)?;

    // Train rotatE
    if cli.train {
        train::spawn_python(&cfg.training, &out)?;
    }

    Ok(())
}
