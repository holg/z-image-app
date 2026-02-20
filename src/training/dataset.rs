//! Training dataset loading for LoRA fine-tuning.
//!
//! Expects a directory containing image files (png, jpg, jpeg, webp)
//! with matching `.txt` caption files.

use std::path::{Path, PathBuf};

use burn::{Tensor, prelude::Backend};
use image::GenericImageView;

/// A single training item: image path + caption.
#[derive(Debug, Clone)]
pub struct TrainingItem {
    pub image_path: PathBuf,
    pub caption: String,
}

/// A training dataset loaded from a directory.
#[derive(Debug, Clone)]
pub struct TrainingDataset {
    pub items: Vec<TrainingItem>,
}

impl TrainingDataset {
    /// Load a dataset from a directory.
    ///
    /// Scans for image files and expects a matching `.txt` file for each.
    /// Returns an error if any image lacks a caption file.
    pub fn from_directory(dir: &Path) -> Result<Self, String> {
        if !dir.is_dir() {
            return Err(format!("Not a directory: {}", dir.display()));
        }

        let image_extensions = ["png", "jpg", "jpeg", "webp"];
        let mut items = Vec::new();

        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory: {e}"))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase());

            if let Some(ext) = ext {
                if image_extensions.contains(&ext.as_str()) {
                    // Look for matching .txt file
                    let txt_path = path.with_extension("txt");
                    if !txt_path.exists() {
                        return Err(format!(
                            "Missing caption file for image: {}",
                            path.display()
                        ));
                    }

                    let caption = std::fs::read_to_string(&txt_path)
                        .map_err(|e| format!("Failed to read caption {}: {e}", txt_path.display()))?
                        .trim()
                        .to_string();

                    if caption.is_empty() {
                        return Err(format!(
                            "Empty caption file: {}",
                            txt_path.display()
                        ));
                    }

                    items.push(TrainingItem {
                        image_path: path,
                        caption,
                    });
                }
            }
        }

        if items.is_empty() {
            return Err(format!(
                "No image+caption pairs found in {}",
                dir.display()
            ));
        }

        // Sort for deterministic ordering
        items.sort_by(|a, b| a.image_path.cmp(&b.image_path));

        Ok(TrainingDataset { items })
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Load an image from disk and convert to a tensor.
///
/// Returns a tensor of shape `[1, 3, height, width]` in range `[-1, 1]`.
pub fn load_image_tensor<B: Backend>(
    path: &Path,
    target_width: u32,
    target_height: u32,
    device: &B::Device,
) -> Result<Tensor<B, 4>, String> {
    let img = image::open(path)
        .map_err(|e| format!("Failed to load image {}: {e}", path.display()))?;

    // Resize to target dimensions
    let img = img.resize_exact(
        target_width,
        target_height,
        image::imageops::FilterType::Lanczos3,
    );

    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let raw = rgb.as_raw();

    // Convert to [1, 3, H, W] tensor normalized to [-1, 1]
    let mut data = vec![0.0f32; 3 * (h as usize) * (w as usize)];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let idx = (y * w as usize + x) * 3;
            for c in 0..3 {
                let pixel = raw[idx + c] as f32 / 255.0;
                let normalized = pixel * 2.0 - 1.0; // [0,1] -> [-1,1]
                data[c * (h as usize) * (w as usize) + y * (w as usize) + x] = normalized;
            }
        }
    }

    let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), device)
        .reshape([1, 3, h as usize, w as usize]);

    Ok(tensor)
}
