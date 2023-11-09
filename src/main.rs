extern crate tch;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor, vision};

const BATCH_SIZE: i64 = 64;
const VALIDATION_BATCH_SIZE: i64 = 1000;
const EPOCHS: i64 = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Set device to CPU or CUDA
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    // Load the MNIST dataset
    let mnist = vision::mnist::load_dir("data")?;
    // let train_images = mnist.train_images.to_device(device);
    // let train_labels = mnist.train_labels.to_device(device);
    // let test_images = mnist.test_images.to_device(device);
    // let test_labels = mnist.test_labels.to_device(device);

    // // Split the training data into training and validation sets
    // let num_samples = train_images.size()[0];
    // let validation_size = (num_samples as f64 * 0.2) as i64; // 20% for validation
    // let train_size = num_samples - validation_size;

    // let train_dataset = Tensor::narrow(&train_images, 0, 0, train_size);
    // let train_labels = Tensor::narrow(&train_labels, 0, 0, train_size);
    // let validation_dataset = Tensor::narrow(&train_images, 0, train_size, validation_size);
    // let validation_labels = Tensor::narrow(&train_labels, 0, train_size, validation_size);
    
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

    // Split dataset into training and validation sets
    let (train_dataset, validation_dataset) = mnist.train_dataset().split_with_ratio(0.8);
    let train_dl = train_dataset.to_data_loader(vs.device(), BATCH_SIZE);
    let validation_dl = validation_dataset.to_data_loader(vs.device(), VALIDATION_BATCH_SIZE);

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
            let predictions = net.forward(&bimages).argmax1(-1, false);
            total_correct += predictions.eq1(&blabels).sum_int();
            total += blabels.size()[0];
        }
        let accuracy = total_correct as f64 / total as f64;
        println!("Epoch: {}, Validation accuracy: {:.2}%", epoch, accuracy * 100.0);
    }

    Ok(())
}
