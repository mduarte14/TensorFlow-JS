import fs from "fs";
import path from "path";
import * as tf from "@tensorflow/tfjs-node-gpu"; // Using the GPU version
import sharp from "sharp"; // Import sharp for image blurring and augmentations

console.log(tf.getBackend());

// Path to the Lorcana images folder
const imageFolder = path.join(process.cwd(), "lorcana_images");
const cardDataPath = path.join(process.cwd(), "sanitizedAllCards.json");

function loadCardData() {
  const cardData = fs.readFileSync(cardDataPath, "utf8");
  return JSON.parse(cardData).cards;
}

// Apply less aggressive augmentations to keep the model focused on the important parts
async function applyImageAugmentations(imageBuffer) {
  let augmentedImage = sharp(imageBuffer);

  if (Math.random() > 0.5) augmentedImage = augmentedImage.flip(); // Horizontal flip
  if (Math.random() > 0.5) augmentedImage = augmentedImage.flop(); // Vertical flip

  const rotationDegrees = Math.random() * 20 - 10; // Subtle rotation range
  augmentedImage = augmentedImage.rotate(rotationDegrees);

  // Resize to 224x224
  augmentedImage = augmentedImage.resize({ width: 224, height: 224 });

  // Slight brightness and contrast adjustments
  augmentedImage = augmentedImage.modulate({
    brightness: 1.05,
    contrast: 1.05,
  });

  const finalBuffer = await augmentedImage.toBuffer();
  return finalBuffer;
}

// Load, augment, and preprocess images
async function loadAndPreprocessImage(imagePath) {
  let imageBuffer = fs.readFileSync(imagePath);

  // Apply augmentations
  imageBuffer = await applyImageAugmentations(imageBuffer);

  const imageTensor = tf.node.decodeImage(imageBuffer, 3);
  const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);
  const normalizedImage = resizedImage.div(255);
  imageTensor.dispose();
  return normalizedImage;
}

// Create dataset from images and assign sequential labels
async function createDataset() {
  const cardData = loadCardData();

  let imageFiles = fs.readdirSync(imageFolder).slice(0, 1195); // Limit to first 1195 images

  // Sort the image files numerically (1.jpg, 2.jpg, ..., 1195.jpg)
  imageFiles = imageFiles.sort((a, b) => {
    const numA = parseInt(a.split(".")[0], 10);
    const numB = parseInt(b.split(".")[0], 10);
    return numA - numB;
  });

  const dataset = [];
  const labels = [];

  // Ensure we only process up to 1195 images (no off-by-one)
  for (let i = 0; i < imageFiles.length; i++) {
    const imageFile = imageFiles[i];
    const imageId = (i + 1).toString(); // Sequential labels (1, 2, ..., 1195)

    const imagePath = path.join(imageFolder, imageFile);
    const imageTensor = await loadAndPreprocessImage(imagePath);
    dataset.push(imageTensor);
    labels.push(parseInt(imageId, 10)); // Ensure label is a number
  }

  return { dataset, labels };
}

// Create a simpler model to reduce overfitting
function createModel(numClasses) {
  const regularizer = tf.regularizers.l2({ l2: 0.03 }); // Increased regularization strength

  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [224, 224, 3],
      filters: 16, // Reduced filters
      kernelSize: 3,
      activation: "relu",
      kernelRegularizer: regularizer,
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.dropout(0.6)); // Increased dropout

  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      kernelRegularizer: regularizer,
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.dropout(0.4));

  model.add(tf.layers.flatten());
  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
      kernelRegularizer: regularizer,
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout(0.5));

  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));

  const optimizer = tf.train.adam(0.0003); // Reduced learning rate

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

  const uniqueLabels = Array.from(new Set(labels)); // Ensure unique labels
  const labelToIndex = Object.fromEntries(
    uniqueLabels.map((label, index) => [label, index])
  );

  for (let i = 0; i < dataset.length; i++) {
    imageTensors.push(dataset[i]);
    labelIndices.push(labelToIndex[labels[i]]); // Map labels to indices
  }

  const xTrain = tf.stack(imageTensors);
  const yTrain = tf.oneHot(
    tf.tensor1d(labelIndices, "int32"),
    uniqueLabels.length
  );

  return { xTrain, yTrain, uniqueLabels };
}

// Train the model
async function trainModel() {
  const { dataset, labels } = await createDataset();
  const numClasses = Array.from(new Set(labels)).length;
  const model = createModel(numClasses);
  const { xTrain, yTrain } = prepareDataForTraining(dataset, labels);

  console.log("Starting training...");

  const earlyStoppingCallback = tf.callbacks.earlyStopping({
    monitor: "val_loss",
    patience: 3,
  });

  await model.fit(xTrain, yTrain, {
    epochs: 5,
    batchSize: 16, // Slightly reduced batch size
    validationSplit: 0.3,
    callbacks: [earlyStoppingCallback],
  });

  console.log("Training complete. Saving model...");
  await model.save("file://./lorcana-model");
  console.log("Model saved successfully.");
}

trainModel().catch(console.error);
