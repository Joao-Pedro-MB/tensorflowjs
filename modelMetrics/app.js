document.addEventListener("DOMContentLoaded", () => {
    const sensorForm = document.getElementById("sensorForm");
    sensorForm.addEventListener("submit", handleFormSubmit);
  });

  document.addEventListener("DOMContentLoaded", () => {
    const reportsDiv = document.getElementById("reports");

  
    // Simulate battery info on button click
    const simulateButton = document.getElementById("simulateButton");
    simulateButton.addEventListener("click", () => {
      startModel();
    });
  });
  


  
  // plot the data we are working with
  async function plot(pointsArray, featureName) {
    tfvis.render.scatterplot(
        {name: `${featureName} vs stroke diagnosis`},
        {values: [pointsArray], series: ["original"]},
        {
            xLabel: featureName,
            yLabel: "bmi",
        }
    )
  }

  //norlisation funtion
  async function normalise(rawTensor) {
    // Ensure the input tensor contains numeric data
    if (rawTensor.dtype !== 'float32') {
      throw new Error("Input tensor must have 'float32' data type");
    }
  
    const numericTensor = rawTensor;
  
    const min = await numericTensor.min().data();
    const max = await numericTensor.max().data();
  
    const normalizedTensor = numericTensor.sub(min).div(max.sub(min));
    return {
      tensor: normalizedTensor,
      min: min[0], // Convert tensor data to numeric value
      max: max[0], // Convert tensor data to numeric value
    };
  }

  //create model
  function createModel() {
    const model = tf.sequential();
  
    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: 'linear',
        inputDim: 1,
    }))
  
    const optimizer = tf.train.sgd(0.1);
  
    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer,
    })
  
    return model;
  }
  
  //train model
  async function trainModel (model, trainingFeatureTensor, trainingLabelTensor) {
  
    const {onBatchEnd, onEpochEnd} = tfvis.show.fitCallbacks(
        {name: "Training Performance"},
        ['loss']
    )
    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: 32,
        epochs: 2000,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd, onBatchEnd, // onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`),
        }
    });
  }
        
  // main funtion
  async function startModel(selectedSensors) {
    // Load data (needs to be from a URL to work in a browser)
    const strokeDataset = await tf.data.csv("http://localhost:8080/web-server/healthcare-dataset-stroke-data.csv");
  
    // Make a map visualization of the data points with tfvis
    const pointsDataset = strokeDataset.map(record => ({
      x: parseFloat(record.avg_glucose_level), // Convert to float
      y: parseFloat(record.bmi), // Convert to float
    }));
    const points = await pointsDataset.toArray();
  
    // Shuffle data to avoid model bias
    tf.util.shuffle(points);
    if (points.length % 2 != 0) {
      points.pop(); // If odd number of points, remove one to divide the dataset
    }
    plot(points, "Average Glucose Level");
  
    // Features (inputs)
    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1], 'float32'); // Specify data type
  
    // Labels (outputs)
    const labelValues = points.map(p => p.y);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1], 'float32'); // Specify data type
  
    // Normalized features and labels
    const normalisedFeature = await normalise(featureTensor);
    const normalisedLabel = await normalise(labelTensor);

    const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalisedFeature.tensor, 2);
    const [trainingLabelTensor, testingLabelTensor] = tf.split(normalisedLabel.tensor, 2);


    // create the model and show its infos
    const model = createModel();
    tfvis.show.modelSummary({name: "model summary"}, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({name: "Layer"}, layer);
  
    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
    const trainingLoss = result.history.loss.pop();
    console.log(`Training set loss: ${trainingLoss}`);
    const validationLoss = result.history.val_loss.pop();
    console.log(`Validation set loss: ${validationLoss}`);
  
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    const loss = await lossTensor.dataSync();
    console.log(`Testing set loss: ${loss}`);
  
  }
  
  // anotar dispositivo e dados de hardware junto com browser utilizado.
  // trocar provedor para o ngrok a fimd e publicar portas de acesso
  