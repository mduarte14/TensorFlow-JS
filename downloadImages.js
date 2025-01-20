// Import the required libraries using ES Modules syntax
import fetch from "node-fetch";
import fs from "fs";
import path from "path";

// Path to save the images
const imageSavePath = path.join(process.cwd(), "lorcana_images");

// Create the folder if it doesn't exist
if (!fs.existsSync(imageSavePath)) {
  fs.mkdirSync(imageSavePath, { recursive: true });
}

// Fetch the JSON data
async function fetchCardData() {
  const response = await fetch(
    "https://lorcanajson.org/files/current/en/allCards.json"
  );
  const data = await response.json();
  return data.cards;
}

// Download an image from a given URL and save it locally
async function downloadImage(url, imageName) {
  const res = await fetch(url);
  const buffer = await res.buffer();
  const imagePath = path.join(imageSavePath, imageName);

  // Write the image to the file system
  fs.writeFileSync(imagePath, buffer);
  console.log(`Downloaded and saved: ${imageName}`);
}

// Main function to download all card images
async function downloadCardImages() {
  const cards = await fetchCardData();

  for (const card of cards) {
    const imageUrl = card.images.full; // Get the full image URL
    const cardId = card.id; // Use the card's unique ID as the file name
    const imageName = `${cardId}.jpg`; // Save the image using the unique card ID

    // Download the image
    await downloadImage(imageUrl, imageName);
  }

  console.log("All images downloaded.");
}

// Run the main function
downloadCardImages().catch(console.error);
