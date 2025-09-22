#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>

// Simple Logistic Regression classifier for text data
class LogisticReg {
private:
    // Bias term for the model
    double b = 0.1;

    // Learning rate for gradient descent
    double lr = 0.01;

    // Set of all unique words found in training data
    std::set<std::string> vocabulary;

    // Stores feature vector and label for each training sentence
    // Example: { "set1" → ( [0,1,0,1...], label ) }
    std::unordered_map<std::string, std::pair<std::vector<int>, int>> wordsvec;

    // Vocabulary converted into list form (for indexing words)
    std::vector<std::string> vocablist;

    // Weights for each word in vocabulary
    std::vector<double> weights;

    // Words to ignore during training (stopwords)
    const std::unordered_set<std::string> stopwords = {
        "is", "to", "a", "be", "am", "i", "are", "my", "in", "the", "and", "or"
    };

    // Provides training data: pairs of sentence + label (1=positive, 0=negative)
    std::vector<std::pair<std::string, int>> training_data();

    // Splits a sentence into lowercase words
    std::vector<std::string> tokenizer(const std::string& sentence);

    // Removes stopwords from tokenized sentence
    std::vector<std::string> preprocessing(const std::string& sentence);

public:
    // Build vocabulary and feature vectors
    void logistic_reg();

    // Train logistic regression model using gradient descent
    void model_trainer();

    // Interactive wrapper to train and test model with user input
    void call_logisticreg();

    // Predict sentiment of a given input string
    std::string predictor(const std::string& input);
};

#endif
