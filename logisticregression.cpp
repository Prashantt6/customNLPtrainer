#include "logisticregression.h"
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <unordered_set>
#include <iterator>

// Training dataset: pairs of sentence + label
// 1 → positive sentiment, 0 → negative sentiment
std::vector<std::pair<std::string, int>> LogisticReg::training_data() {
    std::vector<std::pair<std::string, int>> data{
        {"I love nepal", 1},
        {"I hate cpp", 0},
        {"I live in nepal", 1},
        {"I love my country", 1},
        {"cpp is good", 1},
        {"cpp is a programming language", 1},
        {"I hate eating vegetables", 0},
        {"cpp is sometimes boring", 0},
        {"I like you", 1},
        {"I dislike you", 0},
        {"I am not interested in talking", 0},
        {"I like nepal", 1}
    };
    return data;
}

// Tokenize a sentence into lowercase words
std::vector<std::string> LogisticReg::tokenizer(const std::string& sentence) {
    std::istringstream iss(sentence);
    std::string temp;
    std::vector<std::string> tokens;

    while (iss >> temp) {
        for (char& c : temp) { c = std::tolower(c); } // convert to lowercase
        if (!temp.empty()) {
            tokens.push_back(temp);
        }
    }
    return tokens;
}

// Remove stopwords from tokenized words
std::vector<std::string> LogisticReg::preprocessing(const std::string& sentence) {
    std::vector<std::string> words = tokenizer(sentence);
    std::vector<std::string> result;

    for (const auto& word : words) {
        if (stopwords.find(word) == stopwords.end()) {
            result.push_back(word);
        }
    }
    return result;
}

// Build vocabulary and feature vectors from training data
void LogisticReg::logistic_reg() {
    std::vector<std::pair<std::string, int>> training_set = training_data();
    int set_num = 1;

    // Step 1: Build vocabulary from all training sentences
    for (auto &data : training_set) {
        std::vector<std::string> words = preprocessing(data.first);
        for (auto &word : words) {
            vocabulary.insert(word);
        }
    }

    // Step 2: Convert vocabulary (set) to a list for indexing
    vocablist.assign(vocabulary.begin(), vocabulary.end());

    // Step 3: Create feature vectors for each training sentence
    for (auto &data : training_set) {
        std::vector<std::string> words = preprocessing(data.first);
        std::vector<int> temp_vec(vocablist.size(), 0);

        // Mark word presence (binary features)
        for (auto &word : words) {
            auto it = std::find(vocablist.begin(), vocablist.end(), word);
            if (it != vocablist.end()) {
                int index = std::distance(vocablist.begin(), it);
                temp_vec[index] = 1;
            }
        }

        int label = data.second;
        wordsvec["set" + std::to_string(set_num)] = std::make_pair(temp_vec, label);
        set_num++;
    }
}

// Train logistic regression model using gradient descent
void LogisticReg::model_trainer() {
    weights.assign(vocablist.size(), 0.0);  // initialize weights
    b = 0.1;  // bias term

    for (int epoch = 0; epoch < 1000; epoch++) {
        double total_loss = 0.0;

        // Go through all training examples
        for (auto &sentence : wordsvec) {
            std::vector<int> temp = sentence.second.first;
            int label = sentence.second.second;
            double wx = 0.0;

            // Compute weighted sum
            for (int i = 0; i < weights.size(); i++) {
                wx += weights[i] * temp[i];
            }

            double z = wx + b;

            // Prevent numerical instability
            if (z > 30.0) z = 30.0;
            if (z < -30.0) z = -30.0;

            // Sigmoid activation
            double y = 1.0 / (1.0 + std::exp(-z));

            // Update weights and bias (gradient descent step)
            for (int i = 0; i < weights.size(); i++) {
                weights[i] += lr * (label - y) * temp[i];
            }
            b += lr * (label - y);

            // Cross-entropy loss (for monitoring)
            double loss = -(label * std::log(y + 1e-10) + (1 - label) * std::log(1 - y + 1e-10));
            total_loss += loss;
        }

        // Print average loss every 100 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / wordsvec.size() << std::endl;
        }
    }
}

// Predict sentiment of a new sentence
std::string LogisticReg::predictor(const std::string& input) {
    std::vector<std::string> words = preprocessing(input);
    std::vector<int> x(vocablist.size(), 0);

    // Create feature vector for input
    for (auto &word : words) {
        auto it = std::find(vocablist.begin(), vocablist.end(), word);
        if (it != vocablist.end()) {
            int index = std::distance(vocablist.begin(), it);
            x[index] = 1;
        }
    }

    // Compute prediction
    double wx = 0.0;
    for (int i = 0; i < weights.size(); i++) {
        wx += x[i] * weights[i];
    }

    double z = wx + b;
    double y = 1.0 / (1.0 + std::exp(-z));

    // Threshold at 0.5 → positive or negative
    if (y > 0.5) {
        return "positive";
    } else {
        return "negative";
    }
}

// Wrapper function: train model and allow interactive prediction
void LogisticReg::call_logisticreg() {
    std::string input;
    logistic_reg();   // build features
    model_trainer();  // train model

    std::cout << "Model training completed. Vocabulary size: " << vocablist.size() << std::endl;

    // Keep taking user input until exit
    while (true) {
        std::cout << "Enter a sentence (type 'exit' to quit): ";
        std::getline(std::cin, input);
        if (input == "exit") break;

        std::string prediction = predictor(input);
        std::cout << "Prediction: " << prediction << std::endl;
    }
}
