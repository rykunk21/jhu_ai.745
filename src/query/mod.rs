//! This represents all the code required to perform the query over the learned artifacts

use anyhow::Result;
use ndarray::{Array2, Array3};
use ndarray_npy::read_npy;
use std::collections::HashMap;
use std::path::Path;

// this type is specfic to queries
struct RotatE {
    re: Array2<f64>,
    im: Array2<f64>,
    r_cos: Array2<f64>,
    r_sin: Array2<f64>,
    dim: usize,
}

/// This runs the query over the learned artifacts.
pub fn run(query: &str, model_dir: &Path, data_dir: &Path, alpha: f64, top: usize) -> Result<()> {
    // read in the available entities and their relations
    let ents = load_names(&data_dir.join("entities.dict"));
    let rels = load_names(&data_dir.join("relations.dict"));

    // make a lookup from name of the query to its respective index with a simple hashmap
    let e2i: HashMap<String, usize> = ents
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let r2i: HashMap<String, usize> = rels
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();

    // read in the learned rotatE model and construct the rotatE struct
    let model = load_model(model_dir, ents.len(), rels.len())?;
    let d = compute_distances(&model);
    let q = parse_query(query, &e2i, &r2i)?;
    // display everthing to the terminal
    print_results(&d, &ents, &rels, q, alpha, top);
    Ok(())
}

/// Helper to load the entities and relations from disk. These are just text files with id\tname on each line.
fn load_names(path: &Path) -> Vec<String> {
    let mut names: Vec<Option<String>> = Vec::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let parts: Vec<&str> = line.trim().split('\t').collect();
        if parts.len() != 2 {
            continue;
        }
        let id: usize = parts[0].parse().unwrap();
        if id >= names.len() {
            names.resize(id + 1, None);
        }
        names[id] = Some(parts[1].to_string());
    }
    names.into_iter().map(|n| n.unwrap()).collect()
}

/// loads the learned model from disk and construct the RotatE struct. This is a bit of a pain because the learned model is stored in numpy format, so we have to read it in and convert it to rust types.
/// Thankfully, ndarray_npy has a convienence function I found online
fn load_model(dir: &Path, n_ent: usize, n_rel: usize) -> Result<RotatE> {
    // read the model that we specify in the config with basic protection
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    if cfg.get("model").and_then(|v| v.as_str()) != Some("RotatE") {
        panic!("model must be RotatE");
    }

    // we have to shape up the numpy arrays and rustify them.
    let dim = cfg["hidden_dim"].as_u64().unwrap() as usize;
    let gamma = cfg["gamma"].as_f64().unwrap();
    let e: Array2<f32> = read_npy(&dir.join("entity_embedding.npy")).unwrap();
    let r: Array2<f32> = read_npy(&dir.join("relation_embedding.npy")).unwrap();

    // consumers of these types expect f64
    let e = e.mapv(|v| v as f64);
    let r = r.mapv(|v| v as f64);

    // add some protection to make sure that every relation has 2 entities.
    if e.shape() != &[n_ent, 2 * dim] {
        panic!("entity embedding shape mismatch");
    }
    if r.shape() != &[n_rel, dim] {
        panic!("relation embedding shape mismatch");
    }
    // unflatten and construct the struct
    let re = e.slice(ndarray::s![.., ..dim]).to_owned();
    let im = e.slice(ndarray::s![.., dim..]).to_owned();
    let phase = r / ((gamma + 2.0) / (dim as f64) / std::f64::consts::PI);
    Ok(RotatE {
        re,
        im,
        r_cos: phase.mapv(|v| v.cos()),
        r_sin: phase.mapv(|v| v.sin()),
        dim,
    })
}

/// Computes the pythagorean distance between all possible triplets
fn compute_distances(m: &RotatE) -> Array3<f64> {
    let n = m.re.nrows();
    let nr = m.r_cos.nrows();
    let d = m.dim;
    let mut dist = Array3::<f64>::zeros((n, nr, n));

    // TODO: optimize this for something else besides nested loops. Might be able to index using lsh or hnsw?
    for r in 0..nr {
        for h in 0..n {
            for t in 0..n {
                let mut s = 0.0;
                for k in 0..d {
                    let qre = m.re[[h, k]] * m.r_cos[[r, k]] - m.im[[h, k]] * m.r_sin[[r, k]];
                    let qim = m.re[[h, k]] * m.r_sin[[r, k]] + m.im[[h, k]] * m.r_cos[[r, k]];
                    s += ((qre - m.re[[t, k]]).powi(2) + (qim - m.im[[t, k]]).powi(2)).sqrt();
                }
                dist[[h, r, t]] = s;
            }
        }
    }
    dist
}

/// helper to parse the query so I can specify a "head relation tail" with ? for unknowns.
fn parse_query(
    q: &str,
    e2i: &HashMap<String, usize>,
    r2i: &HashMap<String, usize>,
) -> Result<(Option<usize>, Option<usize>, Option<usize>)> {
    let t: Vec<&str> = q.split_whitespace().collect();
    if t.len() != 3 {
        panic!("query must be 3 tokens");
    }
    let mut v = [None; 3];
    let maps: [&HashMap<String, usize>; 3] = [e2i, r2i, e2i];
    for i in 0..3 {
        if t[i] != "?" {
            v[i] = Some(
                *maps[i]
                    .get(t[i])
                    .ok_or_else(|| anyhow::anyhow!("unknown: {}", t[i]))?,
            );
        }
    }
    let n = v.iter().filter(|x| x.is_none()).count();
    if n == 0 || n > 2 {
        panic!("query needs 1-2 ?");
    }
    Ok((v[0], v[1], v[2]))
}

/// Prints the results to the console-
fn print_results(
    d: &Array3<f64>,
    ents: &[String],
    rels: &[String],
    q: (Option<usize>, Option<usize>, Option<usize>),
    alpha: f64,
    top: usize,
) {
    let (h, r, t) = q;
    let (ne, nr, _) = (d.shape()[0], d.shape()[1], d.shape()[2]);
    let mut cands: Vec<(f64, Vec<usize>)> = Vec::new(); // container for candidates

    // collect the possible patterns, determining which are "?"
    match (h, r, t) {
        (Some(hv), Some(rv), None) => {
            for tv in 0..ne {
                cands.push((d[[hv, rv, tv]], vec![tv]));
            }
        }
        (Some(hv), None, Some(tv)) => {
            for rv in 0..nr {
                cands.push((d[[hv, rv, tv]], vec![rv]));
            }
        }
        (None, Some(rv), Some(tv)) => {
            for hv in 0..ne {
                cands.push((d[[hv, rv, tv]], vec![hv]));
            }
        }
        (Some(hv), None, None) => {
            for rv in 0..nr {
                for tv in 0..ne {
                    cands.push((d[[hv, rv, tv]], vec![rv, tv]));
                }
            }
        }
        (None, Some(rv), None) => {
            for hv in 0..ne {
                for tv in 0..ne {
                    cands.push((d[[hv, rv, tv]], vec![hv, tv]));
                }
            }
        }
        (None, None, Some(tv)) => {
            for hv in 0..ne {
                for rv in 0..nr {
                    cands.push((d[[hv, rv, tv]], vec![hv, rv]));
                }
            }
        }
        _ => {}
    }

    // compute basic statistics and probabilities
    let max_d = cands
        .iter()
        .map(|(d, _)| *d)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = cands
        .iter()
        .map(|(d, _)| (-alpha * (d - max_d)).exp())
        .sum();

    // translate into softmax so we have probabilities
    let mut probs: Vec<(usize, f64)> = cands
        .iter()
        .enumerate()
        .map(|(i, (d, _))| (i, (-alpha * (d - max_d)).exp() / sum))
        .collect();

    // Total shannon entropy of the distribution
    let entropy = -probs.iter().map(|(_, p)| p * p.ln()).sum::<f64>();

    // print eveything nicely so its readable
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let shown = [
        h.map(|v| ents[v].clone())
            .unwrap_or_else(|| "?".to_string()),
        r.map(|v| rels[v].clone())
            .unwrap_or_else(|| "?".to_string()),
        t.map(|v| ents[v].clone())
            .unwrap_or_else(|| "?".to_string()),
    ]
    .join(" ");
    println!("query: {}", shown);
    println!("entropy: {:.4}", entropy);

    println!("{:<5}{:<10}{:<9}{}", "rank", "prob", "dist", "candidate");
    let unknown: Vec<usize> = [0, 1, 2]
        .into_iter()
        .filter(|&i| match i {
            0 => q.0.is_none(),
            1 => q.1.is_none(),
            2 => q.2.is_none(),
            _ => false,
        })
        .collect();
    for (rank, &(idx, prob)) in probs.iter().take(top).enumerate() {
        let mut full = [h, r, t];
        for (a, &c) in unknown.iter().zip(cands[idx].1.iter()) {
            full[*a] = Some(c);
        }
        let label = unknown
            .iter()
            .map(|&a| match a {
                0 => ents[full[0].unwrap()].clone(),
                1 => rels[full[1].unwrap()].clone(),
                2 => ents[full[2].unwrap()].clone(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "{:<5}{:<10.4}{:<9.3}{}",
            rank + 1,
            prob,
            cands[idx].0,
            label
        );
    }
}
