//! LoRA weight save/load utilities.
//!
//! Saves only the LoRA A/B matrices (not the full base model) to a safetensors file,
//! with a JSON sidecar containing the LoRA configuration (rank, alpha, targets).
//!
//! File layout for a LoRA named "my_lora":
//! - `my_lora.safetensors` — LoRA A/B weight tensors only
//! - `my_lora.lora.json`   — LoRA config (rank, alpha, which layers were targeted)

use std::path::{Path, PathBuf};

use burn::{
    prelude::Backend,
    store::{ModuleSnapshot, ModuleStore, SafetensorsStore},
};
use z_image::modules::transformer::lora_transformer::{LoraConfig, ZImageModelLora};

/// Save LoRA weights (only A/B matrices) to a safetensors file.
///
/// Creates two files:
/// - `{path}` — the safetensors file with LoRA weight tensors
/// - `{path}.lora.json` — JSON sidecar with LoRA config metadata
pub fn save_lora_weights<B: Backend>(
    model: &ZImageModelLora<B>,
    config: &LoraConfig,
    path: impl AsRef<Path>,
) -> Result<(), String> {
    let path = path.as_ref();

    // Save weight tensors (only lora_a and lora_b)
    let mut store = SafetensorsStore::from_file(path)
        .overwrite(true)
        .with_regex(r"lora_[ab]");

    model
        .save_into(&mut store)
        .map_err(|e| format!("Failed to save LoRA weights: {e}"))?;

    // Save config as JSON sidecar
    let config_path = lora_config_path(path);
    let config_json = serde_json::json!({
        "rank": config.rank,
        "alpha": config.alpha,
        "target_attention": config.target_attention,
        "target_feed_forward": config.target_feed_forward,
        "target_refiners": config.target_refiners,
    });
    std::fs::write(&config_path, serde_json::to_string_pretty(&config_json).unwrap())
        .map_err(|e| format!("Failed to write LoRA config: {e}"))?;

    Ok(())
}

/// Load LoRA weights from a safetensors file into an existing LoRA model.
///
/// The model must have been initialized with a compatible LoRA config (same rank and targets).
/// Use [`load_lora_config`] to read the config from the sidecar file first.
pub fn load_lora_weights<B: Backend>(
    model: &mut ZImageModelLora<B>,
    path: impl AsRef<Path>,
) -> Result<(), String> {
    let mut store = SafetensorsStore::from_file(path.as_ref()).allow_partial(true);

    let result = model
        .load_from(&mut store)
        .map_err(|e| format!("Failed to load LoRA weights: {e}"))?;

    if !result.errors.is_empty() {
        return Err(format!(
            "Errors loading LoRA weights: {:?}",
            result.errors
        ));
    }

    Ok(())
}

/// Read the LoRA configuration from the JSON sidecar file.
///
/// Looks for `{path}.lora.json` next to the safetensors file.
/// Returns `None` if the sidecar doesn't exist.
pub fn load_lora_config(path: impl AsRef<Path>) -> Result<Option<LoraConfig>, String> {
    let config_path = lora_config_path(path.as_ref());

    if !config_path.exists() {
        return Ok(None);
    }

    let contents =
        std::fs::read_to_string(&config_path).map_err(|e| format!("Failed to read config: {e}"))?;

    let json: serde_json::Value =
        serde_json::from_str(&contents).map_err(|e| format!("Failed to parse config: {e}"))?;

    let rank = json["rank"]
        .as_u64()
        .ok_or("Missing 'rank' in LoRA config")? as usize;
    let alpha = json["alpha"].as_f64().unwrap_or(rank as f64) as f32;
    let target_attention = json["target_attention"].as_bool().unwrap_or(true);
    let target_feed_forward = json["target_feed_forward"].as_bool().unwrap_or(true);
    let target_refiners = json["target_refiners"].as_bool().unwrap_or(true);

    Ok(Some(LoraConfig {
        rank,
        alpha,
        target_attention,
        target_feed_forward,
        target_refiners,
    }))
}

/// Get the path for the JSON config sidecar given a weights file path.
fn lora_config_path(weights_path: &Path) -> PathBuf {
    let mut config_path = weights_path.to_path_buf();
    let stem = config_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    config_path.set_file_name(format!("{stem}.lora.json"));
    config_path
}
