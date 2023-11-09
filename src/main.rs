extern crate tch;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, vision};

const BATCH_SIZE: i64 = 64;
const VALIDATION_BATCH_SIZE: i64 = 1000;
const EPOCHS: i64 = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set device to CPU or CUDA
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    // Load the MNIST dataset
    let mnist = vision::mnist::load_dir("data")?;

    // Assuming that mnist has a method to provide training data loader directly
    let (train_dl, validation_dl) = mnist.train_iter(BATCH_SIZE).split_with_ratio(0.8);

    let net = nn::seq()
        .add(nn::conv2d(vs.root(), 1, 32, 5, Default::default()))
        .add_fn(|xs| xs.max_pool2d_default(2).relu())
        .add(nn::conv2d(vs.root(), 32, 64, 5, Default::default()))
        .add_fn(|xs| xs.max_pool2d_default(2).relu())
        .add_fn(|xs| xs.view([-1, 64 * 4 * 4]))
        .add(nn::linear(vs.root(), 64 * 4 * 4, 256, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs.root(), 256, 10, Default::default()));
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    for epoch in 1..=EPOCHS {
        for (bimages, blabels) in train_dl.iter() {
            let loss = net
                .forward(&bimages)
                .cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }

        // Validation loop
        let mut total_correct = 0;
        let mut total = 0;
        for (bimages, blabels) in validation_dl.iter() {
            let predictions = net.forward(&bimages).argmax(-1, false);
            total_correct += predictions.eq1(&blabels).sum_int();
            total += blabels.size()[0];
        }
        let accuracy = total_correct as f64 / total as f64;
        println!("Epoch: {}, Validation accuracy: {:.2}%", epoch, accuracy * 100.0);
    }

    Ok(())
}