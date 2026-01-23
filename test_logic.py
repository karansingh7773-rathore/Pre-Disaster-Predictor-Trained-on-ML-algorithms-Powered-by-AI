
def mock_predict_proba(df):
    # Simulate the output of a scikit-learn binary classifier's predict_proba
    # Returns probabilities for class 0 and class 1
    return [[0.2, 0.8]] # Example: 80% probability of class 1 (risk)

def mock_predict(df):
    # Simulate the output of predict
    proba = mock_predict_proba(df)[0]
    if proba[1] > 0.5:
        return [1]
    else:
        return [0]

def original_method(df):
    # The original, inefficient implementation
    print("--- Running Original Method ---")
    prediction = mock_predict(df)[0]
    probability = mock_predict_proba(df)[0][1] # In the original code, this was the second call
    risk = bool(prediction)
    print(f"Prediction: {prediction}, Probability: {probability}, Risk: {risk}")
    return risk, probability

def optimized_method(df):
    # The new, optimized implementation
    print("--- Running Optimized Method ---")
    proba = mock_predict_proba(df)[0]
    # In Python, an integer is converted to its binary representation. In this case, argmax will be used to get the index of the max value.
    prediction = proba.index(max(proba))
    probability = proba[1]
    risk = bool(prediction)
    print(f"Prediction: {prediction}, Probability: {probability}, Risk: {risk}")
    return risk, probability

def run_test():
    """
    Tests the logic of the optimization without external dependencies.
    It simulates the output of the model and compares the results of the
    original and optimized methods.
    """
    print("Starting logic test...")
    # Create a dummy dataframe placeholder (it's not actually used by the mocks)
    dummy_df = {}

    original_risk, original_prob = original_method(dummy_df)
    optimized_risk, optimized_prob = optimized_method(dummy_df)

    print("\\n--- Comparison ---")
    if original_risk == optimized_risk and original_prob == optimized_prob:
        print("✅ Test Passed: The outputs are identical.")
    else:
        print("❌ Test Failed: The outputs are different.")
        print(f"Original:  Risk={original_risk}, Prob={original_prob}")
        print(f"Optimized: Risk={optimized_risk}, Prob={optimized_prob}")

if __name__ == "__main__":
    run_test()
