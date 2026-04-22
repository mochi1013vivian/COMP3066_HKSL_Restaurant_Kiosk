# Food photo assets for realtime UI

Place your food photos in this folder so the Translation box can show the real image when that food word is detected.

Supported filenames (match token names):

- `hamburger.jpg` (or `.jpeg` / `.png` / `.webp`)
- `fries.jpg`
- `hash_brown.jpg`
- `apple_pie.jpg`

The UI automatically loads the first existing file in this order:
1. `.jpg`
2. `.jpeg`
3. `.png`
4. `.webp`

Example:
- detected token `hamburger` -> loads `data/processed/ui_assets/food/hamburger.jpg`

If no image is found, the UI falls back to emoji-only display.
