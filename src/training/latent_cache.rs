//! Latent caching for training.
//!
//! Pre-computes image latents (via VAE encoder) and text embeddings (via Qwen3)
//! so they can be reused across training epochs without keeping the encoder/text model loaded.

use std::path::{Path, PathBuf};

use burn::{Tensor, prelude::Backend, tensor::DType};
use qwen3_burn::{Qwen3Model, Qwen3Tokenizer};
use z_image::modules::ae::AutoEncoder;

use crate::training::dataset::{TrainingDataset, load_image_tensor};

/// A cached training pair: image latent + text embedding.
pub struct CachedItem<B: Backend> {
    /// Image latent from VAE encoder: `[1, 16, H/8, W/8]`
    pub latent: Tensor<B, 4>,
    /// Text embedding from Qwen3: `[1, seq_len, 2560]`
    pub text_embedding: Tensor<B, 3>,
}

/// Pre-computed latent cache for the entire training dataset.
pub struct LatentCache<B: Backend> {
    pub items: Vec<CachedItem<B>>,
}

impl<B: Backend> LatentCache<B> {
    /// Pre-compute all latents and text embeddings.
    ///
    /// This encodes all training images through the VAE encoder and all captions
    /// through the Qwen3 text encoder. After this, the encoder and text model can
    /// be unloaded to free VRAM.
    pub fn precompute(
        dataset: &TrainingDataset,
        autoencoder: &AutoEncoder<B>,
        tokenizer: &Qwen3Tokenizer,
        text_encoder: &Qwen3Model<B>,
        target_width: usize,
        target_height: usize,
        device: &B::Device,
        progress: impl Fn(usize, usize),
    ) -> Result<Self, String> {
        let total = dataset.len();
        let mut items = Vec::with_capacity(total);

        for (i, item) in dataset.items.iter().enumerate() {
            progress(i, total);

            // Encode image to latent
            let image_tensor = load_image_tensor::<B>(
                &item.image_path,
                target_width as u32,
                target_height as u32,
                device,
            )?;
            let latent = autoencoder.encode(image_tensor);

            // Encode caption to text embedding
            let text_embedding = compute_text_embedding(
                &item.caption,
                tokenizer,
                text_encoder,
                device,
            )?;

            items.push(CachedItem {
                latent,
                text_embedding,
            });
        }

        progress(total, total);
        Ok(LatentCache { items })
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Compute text embedding for a single caption using the Qwen3 text encoder.
///
/// This mirrors the logic in z_image::generate_from_text but returns just the embedding.
fn compute_text_embedding<B: Backend>(
    caption: &str,
    tokenizer: &Qwen3Tokenizer,
    text_encoder: &Qwen3Model<B>,
    device: &B::Device,
) -> Result<Tensor<B, 3>, String> {
    use burn::tensor::{Bool, Int};

    let (input_ids_vec, attention_mask_vec) = tokenizer
        .encode_prompt(caption)
        .map_err(|e| format!("Tokenization error: {e}"))?;

    let seq_len = input_ids_vec.len();

    let input_ids = Tensor::<B, 1, Int>::from_data(input_ids_vec.as_slice(), device)
        .reshape([1, seq_len]);
    let attention_mask = Tensor::<B, 1>::from_data(
        attention_mask_vec
            .iter()
            .map(|&b| if b { 1.0f32 } else { 0.0f32 })
            .collect::<Vec<_>>()
            .as_slice(),
        device,
    )
    .greater_elem(0.5)
    .reshape([1, seq_len]);

    let prompt_embedding = text_encoder.encode(input_ids, attention_mask.clone());

    // Extract valid (non-padded) embeddings
    let prompt_embedding =
        z_image::extract_valid_embeddings(prompt_embedding, attention_mask);

    Ok(prompt_embedding)
}
