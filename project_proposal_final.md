## Final proposal

**Project title:** Continuous HKSL Restaurant Ordering Translator for Fast-Food Ordering  
**Project goal:** build a system that lets a deaf customer sign a short fast-food order continuously, detects words one by one from video, assembles them into a correct English sentence, and presents the result clearly to restaurant staff.

This proposal fits the course direction because it uses both hands plus upper-body motion, aims for continuous-feeling recognition rather than isolated static signs only, and focuses on a real accessibility problem in a constrained service setting.

## Scope

The vocabulary should stay intentionally **constrained** so the model remains feasible and reliable for a student project with a small number of signers. The intended submission vocabulary is:

- Ordering words: `i`, `want`
- Quantity words: `one`, `two`, `three`, `four`, `five`
- Food items: `hamburger`, `fries`, `apple_pie`, `hash_brown`
- Connectors / service words: `and`, `with`, `thank_you`

This is a **closed-domain ordering prototype**, not a full sign-language translator.

## Recognition design

The most practical technical approach is a **word-level temporal recognizer** running on sliding windows of live video. Instead of classifying a single frame, the system captures a short sequence of MediaPipe landmarks and feeds the full landmark sequence into a temporal model so motion and body-relative position can be used to distinguish signs.

The pipeline is:

1. Capture live webcam video.
2. Extract MediaPipe landmarks for both hands and selected upper-body pose points.
3. Feed short frame windows into a temporal classifier.
4. Accept a word only when predictions are stable over time.
5. Remove duplicates and apply simple grammar cleanup.
6. Build the final English order sentence for staff display.

Using hands plus upper-body pose is more suitable than hand-only tracking for signs that depend on arm trajectory, elbow position, shoulder-relative location, or overall movement.

## Sentence structure

Since the grammar is fixed, the app should not attempt free-form translation. It should map accepted words into a small set of controlled order patterns such as:

- I want + quantity + item
- I want + quantity + item + and + quantity + item
- Thank you

Example outputs:
- I want two fries.  
- I want one hamburger and one apple pie.  
- Thank you.

This structure makes the system easier to evaluate and much more robust than unconstrained continuous translation.

## Data plan

Because the project uses only a small number of signers, the dataset should be collected as **word-level landmark sequences** with consistent framing and repeated recording sessions. For the final submission, the priority is to ensure each vocabulary item has enough balanced sequence samples rather than trying to expand into a much larger restaurant lexicon.

Recommended collection rules:

- record each word multiple times per signer across multiple sessions,
- keep shoulders, elbows, wrists, and both hands visible,
- save landmark sequences frame by frame,
- keep the collection protocol consistent across labels.

Because only a few signers are available, the system should be presented honestly as a **prototype for constrained HKSL restaurant ordering support**, not a universal production-ready recognizer.

## Realtime app

The realtime demo should behave like a practical ordering assistant rather than just a classifier dashboard.

Recommended screen design:

- Left side: live customer camera view
- Top / center panel: current detected word
- Main output panel: final assembled order sentence in large English text
- Food visual aids: simple food emoji or image cues such as 🍔, 🍟, 🥧
- Confirmation area: order confirmation / reset flow for presentation clarity

To keep the demo stable, the app should include:

- confidence threshold,
- stable-frame threshold,
- duplicate suppression,
- cooldown before re-accepting a word,
- reset logic,
- grammar cleanup before display.

## Why this is strong

This proposal is strong because it solves a **real and specific communication task** instead of trying to solve all of sign language. Fast-food ordering is exactly the kind of structured interaction where a constrained assistive interface is realistic and useful.

It also creates a clear demo story: customer signs an order, the system recognizes the signed words continuously, the words are assembled into an English order sentence, and restaurant staff can read the result immediately.

## Suggestions

I strongly suggest these implementation choices:

- Train the recognizer on landmark sequences, not raw images.
- Keep the vocabulary small and balanced.
- Keep English-only output for the final sentence.
- Present the system as **continuous recognition with fixed grammar support**.
- Treat any extra speech or multimodal experiments as optional extensions, not the core submission.

## Presentation angle

For the presentation, describe it like this:  
“This project is a video-based HKSL fast-food ordering translator for deaf customers. It uses MediaPipe hand and upper-body landmarks, recognizes signed words continuously from video, assembles them into English food-order sentences, and displays the result clearly for restaurant staff.”