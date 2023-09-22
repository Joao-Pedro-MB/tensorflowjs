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
  async function normalise(rawTensor, previousMin = null, previousMax = null) {
    // Ensure the input tensor contains numeric data
    if (rawTensor.dtype !== 'float32') {
      throw new Error("Input tensor must have 'float32' data type");
    }
  
    const numericTensor = rawTensor;
  
    const min = previousMin || numericTensor.min();
    const max = previousMax || numericTensor.max();
  
    const normalizedTensor = numericTensor.sub(min).div(max.sub(min));
    
    return {
      tensor: normalizedTensor,
      min,
      max
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

  let normalisedFeature, normalisedLabel, trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;
  // main funtion
  async function run(selectedSensors) {
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
    featureValues = points.map(p => p.x);
    featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1], 'float32'); // Specify data type
  
    // Labels (outputs)
    labelValues = points.map(p => p.y);
    labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1], 'float32'); // Specify data type
  
    // Normalized features and labels
    normalisedFeature = await normalise(featureTensor);
    normalisedLabel = await normalise(labelTensor);
    featureTensor.dispose();
    labelTensor.dispose();

    [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalisedFeature.tensor, 2);
    [trainingLabelTensor, testingLabelTensor] = tf.split(normalisedLabel.tensor, 2);


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


  async function predict() {
    const predictionInput = parseInt(32); // TODO: get from html input
    if (isNaN(predictionInput)) {
      console.log("Invalid input");
      return;

    } else {
      tf.tidy(() => {
        const inputTensor = tf.tensor1d([predictionInput]);
        const normalizedInput = normalise(inputTensor, normalisedFeature.min, normalisedFeature.max);
        const normalizedOutputTensor = model.predict(normalizedInput.tensor);

        // TODO: denormalize function
        const outputTensor = denormalise(normalizedOutputTensor, normalisedLabel.min, normalisedLabel.max);

        const outputValue = outputTensor.dataSync()[0];
        console.log(`Prediction: ${outputValue}`);
      })
    }
  }

  // TODO: logic to switch between local and server
  async function load() {
    // recover from local
    model = await tf.loadLayersModel('downloads://my-model');

    // recover from server with a GET request
    //model = await tf.loadLayersModel('http://model-server.domain/download')


    // add model summary and impplement this model in training
  }

  async function save() {
    // save locally
    const savedResults = await model.save('downloads://my-model');
    console.log(savedResults);

    // save on server with a POST request with format multipart/form-data
    //await model.save('http://model-server.domain/upload')
  }

  async function test () {
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    const loss = await lossTensor.dataSync();
    console.log(`Testing set loss: ${loss}`);
  }

  async function train() {
    ["train", "test", "predict", "load", "save"].forEach(id => {
      // disable buttons for test
    });
    const model = createModel();
    tfvis.show.modelSummary({ name: "Model summary"}, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({ name: "Layer"}, layer);

    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
    console.log(result);
    const trainingLoss = result.history.loss.pop();
    console.log(`Training set loss: ${trainingLoss}`);
    const validationLoss = result.history.val_loss.pop();
    console.log(`Validation set loss: ${validationLoss}`);

    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    const loss = await lossTensor.dataSync();
    console.log(`Testing set loss: ${loss}`);
  }

  async function toggleVisor () {
    tfvis.visor().toggle();
  }
  // anotar dispositivo e dados de hardware junto com browser utilizado.
  // trocar provedor para o ngrok a fimd e publicar portas de acesso
  
  // adicionar tipo de conectividade, latencia, tipo de rede movel, rtl, fluxo de dados, etc