import {MnistDataset} from './data_mnist';


// Create and train a simple model
async function trainModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [784] }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({ optimizer: 'sgd', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  const dataset = new MnistDataset();
  const mnistData = await dataset.loadData();

  const trainXs = tf.tensor2d(mnistData.trainImages, [mnistData.trainImages.length, 784]);
  const trainYs = tf.oneHot(tf.tensor1d(mnistData.trainLabels, 'int32'), 10);

  await model.fit(trainXs, trainYs, {
    epochs: 10,
    batchSize: 32,
    validationSplit: 0.15,
    callbacks: tf.node.tensorBoard('./tensorboard_logs')
  });

  console.log('Training finished');
}

// Start training when the page is loaded
window.onload = async function() {
  await trainModel();
};
