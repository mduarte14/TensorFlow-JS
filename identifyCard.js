import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";

// Load and preprocess new image
function loadAndPreprocessImage(imagePath) {
  const imageBuffer = fs.readFileSync(imagePath);
  const imageTensor = tf.node.decodeImage(imageBuffer, 3); // Decode image as RGB
  const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]); // Resize to 224x224
  const normalizedImage = resizedImage.div(255); // Normalize to range 0-1
  imageTensor.dispose(); // Dispose of old tensor to avoid memory leaks
  return normalizedImage;
}

// Function to load your custom model and make predictions
async function predictWithCustomModel(imageTensor, uniqueLabels) {
  const modelPath = "file://./lorcana-model/model.json"; // Path to your saved model
  const model = await tf.loadLayersModel(modelPath); // Load the model

  const expandedImageTensor = imageTensor.expandDims(0); // Add batch dimension
  const predictions = model.predict(expandedImageTensor); // Make prediction
  const predictedIndex = predictions.argMax(1).dataSync()[0]; // Get the index of the highest prediction score

  // Get the predicted card ID based on the index
  const predictedLabel = uniqueLabels[predictedIndex]; // Get the predicted card ID

  expandedImageTensor.dispose();
  return predictedLabel;
}

// Function to classify and compare results for all images
async function classifyAllImages() {
  const imageFolder = path.join(process.cwd(), "lorcana_images"); // Path to your image folder
  let imageFiles = fs.readdirSync(imageFolder).slice(0, 1195); // Limit to the first 1195 images

  // Sort the image files numerically (1.jpg, 2.jpg, ..., 1195.jpg)
  imageFiles = imageFiles
    .filter((file) => file.endsWith(".jpg")) // Ensure only jpg files are processed
    .sort((a, b) => {
      const numA = parseInt(a.split(".")[0], 10);
      const numB = parseInt(b.split(".")[0], 10);
      return numA - numB;
    });

  // Check that imageFiles array has enough valid files to avoid index out of bounds
  if (imageFiles.length < 1195) {
    console.error(
      `Error: Expected 1195 images, but found ${imageFiles.length}. Please check the image folder.`
    );
    return;
  }

  // Create a simple label array [1, 2, 3, ..., 1195]
  const uniqueLabels = Array.from({ length: 1195 }, (_, i) =>
    (i + 1).toString()
  );

  let correctPredictions = 0;
  let totalPredictions = 0;

  // Iterate over 1195 images and compare with predicted labels
  for (let i = 0; i < 1195; i++) {
    const imageFile = imageFiles[i];
    const actualLabel = (i + 1).toString(); // Actual label is simply the index + 1

    const imagePath = path.join(imageFolder, imageFile);

    // Validate that the image file exists
    if (!fs.existsSync(imagePath)) {
      console.warn(`Image file not found: ${imagePath}. Skipping...`);
      continue;
    }

    // Load and preprocess the image
    const imageTensor = loadAndPreprocessImage(imagePath);

    const predictedLabel = await predictWithCustomModel(
      imageTensor,
      uniqueLabels
    ); // Custom model prediction
    console.log(
      `Predicted Label: ${predictedLabel}, Actual Label: ${actualLabel}`
    );

    // Check if prediction is correct
    if (predictedLabel && predictedLabel === actualLabel) {
      correctPredictions++;
    }

    totalPredictions++;
    imageTensor.dispose(); // Dispose tensor to avoid memory leaks
  }

  // Calculate and log accuracy
  const accuracy = (correctPredictions / totalPredictions) * 100;
  console.log(
    `Accuracy: ${accuracy.toFixed(
      2
    )}% (${correctPredictions}/${totalPredictions})`
  );
}

// Run the classification process for the first 1195 images
classifyAllImages().catch(console.error);
