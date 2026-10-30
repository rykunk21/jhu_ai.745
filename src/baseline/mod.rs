//! Baseline module: loads a pretrained causal LM, tokenizes a partial query,
//! runs a forward pass, and returns the probability distribution over the
//! next-token (continuation) vocabulary.

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_transformers::models::quantized_llama::ModelWeights as Llama;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

/// Configuration for the baseline model.
#[derive(Debug, Clone)]
pub struct BaselineConfig {
    /// Hugging Face model repo id containing the GGUF weights, e.g. "TheBloke/Llama-2-7B-GGUF"
    pub model_id: String,
    /// Specific GGUF file inside the repo.
    pub quantized_file: String,
    /// Hugging Face repo containing the tokenizer.json (often the base model).
    /// Defaults to `model_id` if not specified.
    pub tokenizer_repo: Option<String>,
    /// If None, uses CPU. Some("cuda") for GPU, etc.
    pub device: Option<String>,
}

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            model_id: "TheBloke/Llama-2-7B-GGUF".to_string(),
            quantized_file: "llama-2-7b.Q4_K_M.gguf".to_string(),
            tokenizer_repo: Some("NousResearch/Llama-2-7b-hf".to_string()),
            device: None,
        }
    }
}

/// Holds the loaded model, tokenizer, and device so we can run multiple
/// queries without reloading everything.
pub struct BaselineModel {
    model: Llama,
    tokenizer: Tokenizer,
    device: Device,
}

impl BaselineModel {
    /// Load a pretrained causal language model and its tokenizer from Hugging
    /// Face Hub.
    pub fn load(cfg: &BaselineConfig) -> Result<Self> {
        let device = match cfg.device.as_deref() {
            Some("cuda") => Device::new_cuda(0)?,
            Some("metal") => Device::new_metal(0)?,
            _ => Device::Cpu,
        };

        let api = Api::new()?;

        // Load tokenizer — try tokenizer_repo first, then fall back to model_id
        let tokenizer_repo = cfg.tokenizer_repo.as_deref().unwrap_or(&cfg.model_id);
        let tok_repo = api.model(tokenizer_repo.to_string());
        let tokenizer_path = tok_repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        // Load quantized weights (GGUF)
        let weights_repo = api.model(cfg.model_id.clone());
        let gguf_path = weights_repo.get(&cfg.quantized_file)?;
        let mut file = std::fs::File::open(&gguf_path)?;
        let ct = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        let model = Llama::from_gguf(ct, &mut file, &device)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Tokenize a partial query and return the token ids as a `Tensor`.
    fn encode(&self, query: &str) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(query, true)
            .map_err(|e| anyhow!("{e}"))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let ids_i64: Vec<i64> = ids.iter().map(|&v| v as i64).collect();
        Ok(Tensor::new(ids_i64, &self.device)?)
    }

    /// Run a forward pass on `query` and return the probability distribution
    /// over the *next* token (the last position in the sequence).
    ///
    /// Returns a vector of `(token_id, probability)` sorted by descending
    /// probability.
    pub fn next_token_distribution(&mut self, query: &str) -> Result<Vec<(u32, f32)>> {
        // Encode partial query
        let input_ids = self.encode(query)?;
        let input_ids = input_ids.unsqueeze(0)?; // (1, seq_len)

        // Forward pass through the model
        let logits = self.model.forward(&input_ids, 0)?; // (1, seq_len, vocab_size)

        // Extract logits for the *last* position
        let last_logits = logits.get_on_dim(1, logits.dim(1)? - 1)?; // (1, vocab_size)
        let last_logits = last_logits.squeeze(0)?; // (vocab_size,)

        // Compute softmax over the vocabulary
        let probs = softmax_last_dim(&last_logits)?; // (vocab_size,)

        // Convert to Vec<(u32, f32)> and sort descending
        let probs_vec: Vec<f32> = probs.to_vec1()?;
        let mut indexed: Vec<(u32, f32)> = probs_vec
            .into_iter()
            .enumerate()
            .map(|(i, p)| (i as u32, p))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(indexed)
    }

    /// Convenience: return the top-k most likely next tokens with their
    /// probabilities.
    pub fn top_k(&mut self, query: &str, k: usize) -> Result<Vec<(String, f32)>> {
        let dist = self.next_token_distribution(query)?;
        let top = dist.into_iter().take(k);
        let mut result = Vec::with_capacity(k);
        for (id, prob) in top {
            let token_str = self
                .tokenizer
                .decode(&[id], false)
                .map_err(|e| anyhow!("{e}"))?;
            result.push((token_str, prob));
        }
        Ok(result)
    }
}

/// Standalone helper: just run a single query and print the top-k
/// continuations.
pub fn run(query: &str, k: usize, device: Option<String>) -> Result<()> {
    let mut cfg = BaselineConfig::default();
    if let Some(d) = device {
        cfg.device = Some(d);
    }
    let mut model = BaselineModel::load(&cfg)?;
    let top = model.top_k(query, k)?;

    println!("Query: {:?}", query);
    println!("{:<6} {:<20} {}", "rank", "probability", "token");
    for (rank, (token, prob)) in top.iter().enumerate() {
        println!("{:<6} {:<20.6} {:?}", rank + 1, prob, token);
    }
    Ok(())
}
