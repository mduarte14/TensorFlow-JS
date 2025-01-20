import fs from "fs";
import path from "path";
import * as tf from "@tensorflow/tfjs-node-gpu"; // Using the GPU version
console.log(tf.getBackend());
import sharp from "sharp"; // Import sharp for image blurring and augmentations
import readline from "readline"; // Import readline for user input

// Function to prompt user input (yes/no) with timeout
function promptUserInputWithTimeout(query, timeoutMs = 120000) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      rl.close();
      console.log("\nNo response received. Continuing with training...");
      resolve(false); // Default to "no" (continue training)
    }, timeoutMs);

    rl.question(query, (answer) => {
      clearTimeout(timer); // Clear the timer if user responds
      rl.close();
      resolve(answer.toLowerCase() === "yes");
    });
  });
}

// Path to the Lorcana images folder
const imageFolder = path.join(process.cwd(), "lorcana_images");

// Path to the card data (sanitizedAllCards.json)
const cardDataPath = path.join(process.cwd(), "sanitizedAllCards.json");

// Load card data from JSON file
function loadCardData() {
  const cardData = fs.readFileSync(cardDataPath, "utf8");
  return JSON.parse(cardData).cards; // Return the cards array
}

async function applyImageAugmentations(imageBuffer) {
  let augmentedImage = sharp(imageBuffer);

  // Randomly apply horizontal flip
  if (Math.random() > 0.5) {
    augmentedImage = augmentedImage.flip();
  }

  // Randomly apply rotation between -10 and +10 degrees
  const rotationDegrees = Math.random() * 20 - 10;
  augmentedImage = augmentedImage.rotate(rotationDegrees);

  // Resize to 224x224
  augmentedImage = augmentedImage.resize({ width: 224, height: 224 });

  const finalBuffer = await augmentedImage.toBuffer();
  return finalBuffer;
}

// Load, augment, and preprocess images
async function loadAndPreprocessImage(imagePath) {
  let imageBuffer = fs.readFileSync(imagePath);
  const imageTensor = tf.node.decodeImage(imageBuffer, 3); // Decode image as RGB
  const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]); // Resize to 224x224
  const normalizedImage = resizedImage.div(255); // Normalize to range 0-1
  imageTensor.dispose(); // Dispose of old tensor to avoid memory leaks

  return normalizedImage;
}

// Create dataset from images and associate them with their 'id'
async function createDataset() {
  const cardData = loadCardData();
  const imageFiles = fs.readdirSync(imageFolder);
  const dataset = [];
  const labels = [];

  for (const imageFile of imageFiles) {
    const imageId = imageFile.split(".")[0]; // Use card id as the file name
    const card = cardData.find((card) => card.id.toString() === imageId);

    if (card) {
      const imagePath = path.join(imageFolder, imageFile);
      const imageTensor = await loadAndPreprocessImage(imagePath);
      dataset.push(imageTensor);
      labels.push(card.id); // Using the `id` as the label
    } else {
      console.log(`No match found for image: ${imageFile}`);
    }
  }

  return { dataset, labels };
}

// Create the CNN model with reduced complexity and added Dropout layers
function createModel(numClasses) {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [224, 224, 3],
      filters: 128,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.dropout(0.6)); // Dropout layer to prevent overfitting

  model.add(
    tf.layers.conv2d({
      filters: 128,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.dropout(0.25)); // Dropout layer

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout(0.5)); // Dropout layer before final dense layer
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));

  const learningRate = 0.001;
  const optimizer = tf.train.adam(learningRate);

  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

// Prepare data for training
function prepareDataForTraining(dataset, labels) {
  const imageTensors = [];
  const labelIndices = [];
  const uniqueLabels = Array.from(new Set(labels));
  const labelToIndex = Object.fromEntries(
    uniqueLabels.map((label, index) => [label, index])
  );

  for (let i = 0; i < dataset.length; i++) {
    imageTensors.push(dataset[i]);
    labelIndices.push(labelToIndex[labels[i]]);
  }

  const xTrain = tf.stack(imageTensors);
  const yTrain = tf.oneHot(
    tf.tensor1d(labelIndices, "int32"),
    uniqueLabels.length
  );

  return { xTrain, yTrain, uniqueLabels };
}

// Train the model with epochs and a check every 10 epochs to commit progress or continue
async function trainModel() {
  const { dataset, labels } = await createDataset();
  const numClasses = Array.from(new Set(labels)).length;
  const model = createModel(numClasses);
  const { xTrain, yTrain } = prepareDataForTraining(dataset, labels);

  console.log("Starting training...");
  console.time("Training Time");

  for (let epoch = 0; epoch < 50; epoch += 10) {
    // Train for 10 epochs at a time
    await model.fit(xTrain, yTrain, {
      epochs: 10,
      batchSize: 5,
      validationSplit: 0.2,
    });

    console.log(
      `Completed ${epoch + 10} epochs. Would you like to commit the model?`
    );

    const shouldCommit = await promptUserInputWithTimeout(
      "Commit model? (yes/no): "
    );

    if (shouldCommit) {
      console.log("Saving the model...");
      await model.save("file://./lorcana-model");
      console.log("Model saved successfully.");
      break; // Exit if the model is committed
    }
  }

  console.timeEnd("Training Time");
}

// Run the training process
trainModel().catch(console.error);
