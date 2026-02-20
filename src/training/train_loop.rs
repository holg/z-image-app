//! Flow matching training loop for LoRA fine-tuning.

use std::sync::mpsc::{Receiver, Sender};

use burn::{
    Tensor,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{Distribution, backend::AutodiffBackend},
};
use z_image::modules::transformer::lora_transformer::{LoraConfig, ZImageModelLora};

use crate::training::latent_cache::LatentCache;

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub lora_config: LoraConfig,
    pub save_every_n_steps: usize,
    pub output_dir: std::path::PathBuf,
}

/// Messages sent from the training thread to the UI.
#[derive(Debug, Clone)]
pub enum TrainingMessage {
    /// Progress update with current step info.
    Step {
        epoch: usize,
        step: usize,
        total_steps: usize,
        loss: f32,
    },
    /// Training completed successfully.
    Completed,
    /// Training failed with an error.
    Error(String),
    /// Log message.
    Log(String),
}

/// Run the flow matching training loop.
///
/// This is meant to be called from a spawned thread.
///
/// The flow matching objective:
/// - Sample `t ~ Uniform(0, 1)`
/// - Sample `noise ~ N(0, 1)`
/// - `noisy_latents = (1-t) * noise + t * image_latents`
/// - `v_target = image_latents - noise`
/// - `v_predicted = model(noisy_latents, t, text_embedding)`
/// - `loss = MSE(v_predicted, v_target)`
pub fn train<B: AutodiffBackend>(
    mut lora_model: ZImageModelLora<B>,
    latent_cache: &LatentCache<B::InnerBackend>,
    config: &TrainingConfig,
    device: &B::Device,
    tx: Sender<TrainingMessage>,
    cancel_rx: Receiver<()>,
) {
    let total_steps_per_epoch = latent_cache.len();
    let total_steps = config.num_epochs * total_steps_per_epoch;

    tx.send(TrainingMessage::Log(format!(
        "Starting training: {} epochs, {} steps/epoch, {} total steps",
        config.num_epochs, total_steps_per_epoch, total_steps
    )))
    .ok();
    tx.send(TrainingMessage::Log(format!(
        "LoRA params: {}",
        lora_model.lora_param_count()
    )))
    .ok();

    // Create optimizer
    let mut optim = AdamWConfig::new()
        .with_weight_decay(1e-2)
        .init::<B, ZImageModelLora<B>>();

    let mut global_step = 0;

    for epoch in 0..config.num_epochs {
        // Simple sequential iteration (shuffle could be added with rand)
        for (step, item) in latent_cache.items.iter().enumerate() {
            // Check for cancellation
            if cancel_rx.try_recv().is_ok() {
                tx.send(TrainingMessage::Log("Training cancelled.".to_string())).ok();
                tx.send(TrainingMessage::Completed).ok();
                return;
            }

            // Move cached data to autodiff device
            let image_latent: Tensor<B, 4> = Tensor::from_inner(item.latent.clone()).to_device(device);
            let text_embedding: Tensor<B, 3> =
                Tensor::from_inner(item.text_embedding.clone()).to_device(device);

            // Sample random timestep t ~ Uniform(0, 1)
            let t_val: f32 = Tensor::<B, 1>::random([1], Distribution::Uniform(0.0, 1.0), device)
                .into_data()
                .as_slice::<f32>()
                .expect("f32 slice")[0];

            // Sample noise with same shape as latent
            let latent_dims = image_latent.dims();
            let noise: Tensor<B, 4> =
                Tensor::random(latent_dims, Distribution::Normal(0.0, 1.0), device);

            // Create noisy latents: noisy = (1-t) * noise + t * image_latents
            let noisy_latents =
                noise.clone() * (1.0 - t_val) + image_latent.clone() * t_val;

            // Velocity target: v_target = image_latents - noise
            let v_target = image_latent - noise;

            // Forward pass: model predicts velocity
            // The model's forward expects timestep as Tensor<B, 1> in [0, 1]
            // Internally it does: t = 1.0 - timestep, then feeds to timestep embedder
            let timestep = Tensor::<B, 1>::from_floats([t_val], device);
            let v_predicted = lora_model.forward(noisy_latents, timestep, text_embedding);

            // MSE loss
            let diff = v_predicted - v_target;
            let loss = (diff.clone() * diff).mean();

            // Extract loss value for logging (before backward)
            let loss_val = loss
                .clone()
                .into_data()
                .as_slice::<f32>()
                .expect("f32 slice")[0];

            // Backward pass
            let grads = GradientsParams::from_grads(loss.backward(), &lora_model);

            // Optimizer step
            lora_model = optim.step(config.learning_rate, lora_model, grads);

            global_step += 1;

            // Send progress
            tx.send(TrainingMessage::Step {
                epoch,
                step,
                total_steps: total_steps_per_epoch,
                loss: loss_val,
            })
            .ok();
        }

        tx.send(TrainingMessage::Log(format!(
            "Epoch {}/{} complete",
            epoch + 1,
            config.num_epochs
        )))
        .ok();
    }

    tx.send(TrainingMessage::Log("Training complete!".to_string())).ok();
    tx.send(TrainingMessage::Completed).ok();
}
