import fs from "fs";
import path from "path";

// Utility function to clean the fullIdentifier
const cleanIdentifier = (identifier) => {
  return identifier
    .replace(/[\/•\s]+/g, "_") // Replace '/' and '•' with underscores and remove spaces
    .replace(/[^a-zA-Z0-9_]/g, ""); // Remove any other non-alphanumeric characters
};

// Load original allCards.json
const cardDataPath = path.join(process.cwd(), "allCards.json");
const cardData = JSON.parse(fs.readFileSync(cardDataPath, "utf8"));

// Create a new sanitized dataset
const sanitizedCardData = cardData.cards.map((card) => {
  return {
    ...card,
    sanitizedFullIdentifier: cleanIdentifier(card.fullIdentifier), // Add sanitized fullIdentifier
  };
});

// Save the sanitized dataset to a new JSON file
const sanitizedCardDataPath = path.join(
  process.cwd(),
  "sanitizedAllCards.json"
);
fs.writeFileSync(
  sanitizedCardDataPath,
  JSON.stringify({ cards: sanitizedCardData }, null, 2)
);

console.log(`Sanitized dataset saved to ${sanitizedCardDataPath}`);
