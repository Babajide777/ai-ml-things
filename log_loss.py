import math

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss as sklearn_log_loss

# Probability placed on the true answer, from the one-rule table.
TRUTH_PROBABILITIES = [0.9, 0.5, 0.1, 0.01, 0.001]

# P(yes) from the two models when the true answer is no.
MODEL_PROBABILITIES = [0.6, 0.99]


def one_rule_loss(p_truth):
    return -math.log(p_truth)


def main():
    print("One-rule table")
    for p_truth in TRUTH_PROBABILITIES:
        print(f"p_truth={p_truth:<5}  loss={one_rule_loss(p_truth):.2f}")

    print("Models (true label is no)")
    hand_losses = []
    for p_yes in MODEL_PROBABILITIES:
        loss = one_rule_loss(1 - p_yes)
        hand_losses.append(loss)
        print(f"p_yes={p_yes:<4}  loss={loss:.2f}")

    y_true = [0, 0]
    y_pred = MODEL_PROBABILITIES
    hand_average = sum(hand_losses) / len(hand_losses)
    sklearn_loss = sklearn_log_loss(y_true, y_pred, labels=[0, 1])
    print(f"hand average={hand_average:.2f}")
    print(f"sklearn log_loss={sklearn_loss:.2f}")
    print(f"equal={math.isclose(sklearn_loss, hand_average)}")

    thresholded = [int(p >= 0.5) for p in y_pred]
    accuracy = accuracy_score(y_true, thresholded)
    print(f"accuracy at 0.5={accuracy:.2f}")


if __name__ == "__main__":
    main()
