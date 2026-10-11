//! This represents all the code required to perform the training from both extraction by GLiNER and learning the embeddings with rotatE

use crate::config::Config;
use crate::config::TrainingConfig;
use crate::types::Triple;
use anyhow::Result;
use candle_core::{DType, Device};
use gliner_rs::{ExtractOptions, GLiNER2};
use serde_json::Value;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Main run function to extract the schema with GLiNER and setup the rotatE dataset
pub fn run(input: &Path, out: &Path, cfg: &Config) -> Result<()> {
    let text = std::fs::read_to_string(input).unwrap();
    let model = GLiNER2::load(&cfg.model_path, &Device::Cpu, DType::F32)?;
    let schema = gliner_rs::Schema::new()
        .entities(cfg.schema.entities.clone())
        .relations(cfg.schema.relations.clone());
    let mut triples = Vec::new();

    // clean up the text
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let result = model.extract(line, &schema, &ExtractOptions::default())?; // GLiNER extraction
        triples.extend(parse_triples(&result)); // parse it to the rotatE
    }
    // write to disk, so the python rotatE implementation can read it
    write_dataset(out, &triples)?;
    println!("triples: {}", triples.len());
    Ok(())
}

/// Parses triples out from the GLiNER result so its structured and ready to learn with rotatE.
fn parse_triples(v: &Value) -> Vec<Triple> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    // The json output uses this field relation_extraction, just hard code this for now
    let map = match v.get("relation_extraction") {
        Some(Value::Object(o)) => o,
        _ => return out,
    };

    // iterate over all relationship types and their triplets
    for (rel, items) in map {
        let arr = match items {
            Value::Array(a) => a,
            _ => continue,
        };
        // parse the triplet into our tripet struct and add to our triplets container
        for item in arr {
            let (h, t) = extract_pair(item);
            // only add nonempty triplets we havent seen before
            if !h.is_empty()
                && !t.is_empty()
                && h != t
                && seen.insert((h.clone(), rel.clone(), t.clone()))
            {
                out.push(Triple {
                    head: h.to_lowercase(),
                    relation: rel.to_lowercase(),
                    tail: t.to_lowercase(),
                });
            }
        }
    }
    out
}

/// Helper for triplet parsing
fn extract_pair(item: &Value) -> (String, String) {
    if let Value::Array(arr) = item {
        if arr.len() == 2 {
            return (
                arr[0].as_str().unwrap_or("").into(),
                arr[1].as_str().unwrap_or("").into(),
            );
        }
    }
    (get_text(item.get("head")), get_text(item.get("tail")))
}

/// helper for pair extraction
fn get_text(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(o)) => o.get("text").and_then(|v| v.as_str()).unwrap_or("").into(),
        _ => String::new(),
    }
}

/// Write the structured triplets to disk, as well as the entities and relations .dict files that rotatE expects.
fn write_dataset(out: &Path, triples: &[Triple]) -> Result<()> {
    std::fs::create_dir_all(out).unwrap();

    // Make clones of entities and relations to avoid tripping up the borrow checker
    let mut e: Vec<String> = triples
        .iter()
        .flat_map(|t| [t.head.clone(), t.tail.clone()])
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    e.sort();

    let mut r: Vec<String> = triples
        .iter()
        .map(|t| t.relation.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    r.sort();

    // Now just write everything
    let mut f = File::create(out.join("entities.dict")).unwrap();
    for (i, name) in e.iter().enumerate() {
        writeln!(f, "{}\t{}", i, name).unwrap();
    }
    let mut f = File::create(out.join("relations.dict")).unwrap();
    for (i, name) in r.iter().enumerate() {
        writeln!(f, "{}\t{}", i, name).unwrap();
    }
    let mut f = File::create(out.join("train.txt")).unwrap();
    for t in triples {
        writeln!(f, "{}\t{}\t{}", t.head, t.relation, t.tail).unwrap();
    }
    // GLiNER needs these empty files to work
    File::create(out.join("test.txt")).unwrap();
    File::create(out.join("valid.txt")).unwrap();
    Ok(())
}

/// This function runs the rotatE training. We just add all the flags directly that we got from reading the rotatE repo.
pub fn spawn_python(cfg: &TrainingConfig, data_path: &Path) -> Result<()> {
    let mut cmd = std::process::Command::new("python");
    cmd.arg("-u")
        .arg(&cfg.script_path)
        .arg("--do_train")
        .arg("--data_path")
        .arg(data_path)
        .arg("--model")
        .arg(&cfg.model_type)
        .arg("-n")
        .arg(cfg.negative_sample_size.to_string())
        .arg("-b")
        .arg(cfg.batch_size.to_string())
        .arg("-d")
        .arg(cfg.hidden_dim.to_string())
        .arg("-g")
        .arg(cfg.gamma.to_string())
        .arg("-a")
        .arg(cfg.alpha.to_string())
        .arg("-lr")
        .arg(cfg.learning_rate.to_string())
        .arg("--max_steps")
        .arg(cfg.max_steps.to_string())
        .arg("-r")
        .arg(cfg.regularization.to_string())
        .arg("-save")
        .arg(&cfg.save_path)
        .arg("-adv")
        .arg("-de");

    let out = cmd.output()?;
    if !out.stdout.is_empty() {
        println!("{}", String::from_utf8_lossy(&out.stdout));
    }
    if !out.stderr.is_empty() {
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
    }
    if !out.status.success() {
        anyhow::bail!("python exited with {:?}", out.status);
    }
    Ok(())
}
