document.addEventListener("DOMContentLoaded", () => {
  const sensorForm = document.getElementById("sensorForm");
  sensorForm.addEventListener("submit", handleFormSubmit);
});

async function handleFormSubmit(event) {
  event.preventDefault();

  const selectedSensors = [];
  const checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');

  checkboxes.forEach((checkbox) => {
    selectedSensors.push(checkbox.value);
  });

  // Start training the TensorFlow.js model here
  await startModel(selectedSensors);

  // Function to add content to the reports div
  function addToReports(content) {
    const reportsDiv = document.getElementById("reports");
    const paragraph = document.createElement("p");
    paragraph.textContent = content;
    reportsDiv.appendChild(paragraph);
  }

  // Print sensor values based on selected checkboxes
  if (selectedSensors.includes("battery")) {
    printBatteryInfo(addToReports);
  }
  if (selectedSensors.includes("location")) {
    startGeolocation(addToReports);
  }
  if (selectedSensors.includes("gyroscope")) {
    startGyroscope(addToReports);
  }
  if (selectedSensors.includes("auxiliarySensors")) {
    startAccelerometer(addToReports);
    startAmbientLightSensor(addToReports);
    startProximitySensor(addToReports);
  }

  // Print sensor data every 5 seconds
  const interval = setInterval(() => {
    if (selectedSensors.includes("battery")) {
      printBatteryInfo(addToReports);
    }
    if (selectedSensors.includes("location")) {
      startGeolocation(addToReports);
    }
    if (selectedSensors.includes("gyroscope")) {
      startGyroscope(addToReports);
    }
    if (selectedSensors.includes("auxiliarySensors")) {
      startAccelerometer(addToReports);
    }
  }, 5000);

  // Stop printing after 5 minutes (300 seconds)
  setTimeout(() => {
    clearInterval(interval);
  }, 300000); // 5 minutes in milliseconds
}

function printHardwareInfo(addToReports) {
  const memoryUsage = performance.memory;
  const processorInfo = navigator.hardwareConcurrency;

  // Print browser, processor, and memory usage data
  addToReports(`Memory used: ${memoryUsage.usedJSHeapSize} bytes`);
  addToReports(`Memory total: ${memoryUsage.totalJSHeapSize} bytes`);
  addToReports(`Processor cores: ${processorInfo}`);
}

function printBatteryInfo(addToReports) {
  navigator.getBattery().then((battery) => {
    addToReports(`Battery charging? ${battery.charging ? "Yes" : "No"}`);
    addToReports(`Battery level: ${battery.level * 100}%`);
    addToReports(`Battery charging time: ${battery.chargingTime} seconds`);
    addToReports(`Battery discharging time: ${battery.dischargingTime} seconds`);
  });
}

function startGeolocation(addToReports) {
  function success(position) {
    addToReports(`Latitude: ${position.coords.latitude}, Longitude: ${position.coords.longitude}`);
  }

  function error() {
    addToReports("Sorry, no position available.");
  }

  const options = {
    enableHighAccuracy: true,
    maximumAge: 30000,
    timeout: 27000,
  };

  const watchID = navigator.geolocation.watchPosition(success, error, options);
}

function startGyroscope(addToReports) {
  if ('Gyroscope' in window) {
    const gyro = new Gyroscope({ frequency: 60 });

    gyro.addEventListener('reading', () => {
      addToReports('Gyroscope values:');
      addToReports(`x: ${gyro.x}`);
      addToReports(`y: ${gyro.y}`);
      addToReports(`z: ${gyro.z}`);
    });

    gyro.start();
  } else {
    addToReports('Warning: Gyroscope not supported on this device.');
  }
}

function startProximitySensor(addToReports) {
  if ('ProximitySensor' in window) {
    const proximitySensor = new ProximitySensor();

    proximitySensor.addEventListener('reading', () => {
      addToReports(`Proximity: ${proximitySensor.near ? 'Near' : 'Far'}`);
    });

    proximitySensor.start();
  } else {
    addToReports('Proximity sensor not supported on this device.');
  }
}

function startAmbientLightSensor(addToReports) {
  if ('AmbientLightSensor' in window) {
    const ambientLightSensor = new AmbientLightSensor();

    ambientLightSensor.addEventListener('reading', () => {
      addToReports(`Ambient Light Level: ${ambientLightSensor.illuminance} lux`);
    });

    ambientLightSensor.start();
  } else {
    addToReports('Ambient light sensor not supported on this device.');
  }
}

function startAccelerometer(addToReports) {
  let accelerometer = new Accelerometer({ frequency: 60 });
  accelerometer.addEventListener("reading", (e) => {
    addToReports(`Acceleration along the X-axis: ${accelerometer.x}`);
    addToReports(`Acceleration along the Y-axis: ${accelerometer.y}`);
    addToReports(`Acceleration along the Z-axis: ${accelerometer.z}`);
  });
  accelerometer.start();
}


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

//denormalisation funtion
function denormalise(normalisedTensor, min, max) {
          const denormalisedTensor = normalisedTensor.mul(max.sub(min)).add(min);
          return denormalisedTensor;
}

//norlisation funtion
function normalise(rawTensor) {
          const numericTensor = tf.cast(rawTensor, 'float32');

          const min = numericTensor.min();
          const max = numericTensor.max();

          const normalisedTensor = numericTensor.sub(min).div(max.sub(min));
          return {
              tensor: normalisedTensor,
              min: min,  // Convert tensor to numeric value
              max: max,  // Convert tensor to numeric value
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
          // load data (need to be from url to work in browser)
          const strokeDataset = tf.data.csv("http://localhost:8080/web-server/healthcare-dataset-stroke-data.csv");
          const sample = strokeDataset.take(10);
          const dataArray = await sample.toArray();
          console.log(dataArray);

          // make a map visualization of the data points with tfvis
          const pointsDataset = strokeDataset.map(record => ({
              x: record.avg_glucose_level,
              y: record.bmi,
          }))
          const points = await pointsDataset.toArray();

          // shuffle data to avoid model bias
          tf.util.shuffle(points);
          if (points.length % 2 != 0) { // if odd numbers pop one to divide originl dataset in test and training halves
              points.pop()
          }
          plot(points, "Avarage Glugose Level");

          // features (inputs)
          const featureValues = points.map(p => p.x);
          const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

          // labels (outputs)
          const labelValues = points.map(p => p.y);
          const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

          // normalised features and labels
          const normalisedFeature = normalise(featureTensor);
          const normalisedLabel = normalise(labelTensor);

          //denormalise(normalisedFeature.tensor, normalisedFeature.max, normalisedFeature.min).print();
          //denormalise(normalisedLabel.tensor, normalisedLabel.max, normalisedLabel.min).print();

          const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalisedFeature.tensor, 2);
          const [trainingLabelTensor, testingLabelTensor] = tf.split(normalisedFeature.tensor, 2);

          console.log("Training the model with selected sensors:", selectedSensors);

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
