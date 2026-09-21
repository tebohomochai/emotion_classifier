"""
Interactive script for trying the trained emotion classifier.

"""

from model import EmotionClassifier


def main():
    print("Loading model...")
    clf = EmotionClassifier()
    print("Ready. Type a headline to classify it, or 'quit' to exit.\n")

    while True:
        text = input("Headline: ").strip()

        if text.lower() in ("quit", "exit"):
            print("Thansk, bye!")
            break

        if not text:
            continue

        result = clf.predict(text)

        #Sort probabilities highest first, for readability
        sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)

        print(f"  Predicted: {result['label']}")
        for label, prob in sorted_probs:
            print(f"    {label:10s} {prob:.3f}")
        print()


if __name__ == "__main__":
    main()
