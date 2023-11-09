use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor, Device, nn::VarStore, Kind};
use mnist::*;
use tqdm::tqdm;
// use ndarray::prelude::*;


fn main() {


    // Load the data using the MnistBuilder, just like before
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..

    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Convert the flat u8 image data to f32 and then create a tensor
    let train_images_f32: Vec<f32> = trn_img.into_iter().map(|x| x as f32 / 255.0).collect();
    let train_images_tensor = Tensor::of_slice(&train_images_f32)
        .view([-1, 28, 28]);

    // Convert the u8 labels data to i64 and then create a tensor
    let train_labels_i64: Vec<i64> = trn_lbl.into_iter().map(|x| x as i64).collect();
    let train_labels_tensor = Tensor::of_slice(&train_labels_i64);

    // Repeat the same process for test images and labels
    let test_images_f32: Vec<f32> = tst_img.into_iter().map(|x| x as f32 / 255.0).collect();
    let test_images_tensor = Tensor::of_slice(&test_images_f32)
        .view([-1, 28, 28]);

    let test_labels_i64: Vec<i64> = tst_lbl.into_iter().map(|x| x as i64).collect();
    let test_labels_tensor = Tensor::of_slice(&test_labels_i64);

    
    // Define the dimensions of our input and output layers
    let input_size = 784;
    let hidden_size = 128;
    let output_size = 10;

    // Initialize the VarStore with the correct device
    // let vs = VarStore::new(Device::Cuda(0));
    let vs = VarStore::new(Device::Cpu);

    // Define the network architecture using Sequential
    let cfg = nn::LinearConfig::default();
    let mut net = nn::seq();
    net = net.add(nn::linear(vs.root() / "layer1", input_size, hidden_size, cfg));
    net = net.add_fn(|xs| xs.relu());
    net = net.add(nn::linear(vs.root() / "layer2", hidden_size, output_size, cfg));
    
    // Define the optimizer
    let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();

    let mut last_batch_index = 0;
    let mut last_loss_float = 0.0;

    // Define the number of epochs and the batch size
    let num_epochs = 10; // for example
    let batch_size = 64; // for example
    let num_batches = train_images_tensor.size()[0] / batch_size;

    // Training loop for multiple epochs
    for epoch in tqdm(1..=num_epochs) {
        for batch_index in tqdm(0..num_batches) {
            // println!("Epoch: {}, Batch: {}", epoch, batch_index);
            let batch_tensor = train_images_tensor.narrow(0, batch_index * batch_size, batch_size);
            let labels_tensor = train_labels_tensor.narrow(0, batch_index * batch_size, batch_size);
            // println!("batch size: {:?}", batch_tensor.size());
            let batch = batch_tensor.view([-1, 28 * 28]);
            let labels = labels_tensor;
            // println!("batch size: {:?}", batch.size());
            // Forward pass
            let predictions = net.forward(&batch);
            let loss = predictions.cross_entropy_for_logits(&labels);

            // Backward pass and optimizer step
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // Print the loss
            let loss_float: f64 = loss.double_value(&[]);
            // println!("Epoch: {}, Batch: {}, Loss: {}", epoch, batch_index, loss_float);
            last_batch_index = batch_index;
            last_loss_float = loss_float;
        }
        println!("Epoch: {}, Batch: {}, Loss: {}", epoch, last_batch_index, last_loss_float);
    }
    // Assuming the test dataset is large and needs to be processed in batches
    let test_batch_size = 64; // for example
    let num_test_batches = test_images_tensor.size()[0] / test_batch_size;
    let mut total_correct = 0;
    let mut total = 0;

    for batch_index in 0..num_test_batches {
        let batch_tensor = test_images_tensor.narrow(0, batch_index * test_batch_size, test_batch_size);
        let labels_tensor = test_labels_tensor.narrow(0, batch_index * test_batch_size, test_batch_size);

        let batch = batch_tensor.view([-1, 28 * 28]);
        let labels = labels_tensor;

        // Forward pass without gradient tracking since we're in test mode
        let logits = net.forward(&batch);
        let predicted_labels = logits.argmax(-1, false);
        let correct_predictions = predicted_labels.eq_tensor(&labels); // Remove the reference for labels
        let num_correct = correct_predictions.to_kind(Kind::Int64).sum(Kind::Int64);
        total_correct += num_correct.int64_value(&[]); // Correctly extract the scalar value
        total += test_batch_size;
    }

    let test_accuracy = (total_correct as f64) / (total as f64);
    println!("Test Accuracy: {:?}", test_accuracy);
}