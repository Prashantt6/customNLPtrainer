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

std::vector<std::string> LogisticReg::tokenizer(const std::string& sentence) {
    std::istringstream iss(sentence);
    std::string temp;
    std::vector<std::string> tokens;
    while (iss >> temp) {
        for (char& c : temp) { c = std::tolower(c); }
        if (!temp.empty()) {
            tokens.push_back(temp);
        }
    }
    return tokens;
}

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

void LogisticReg::logistic_reg() {
    std::vector<std::pair<std::string, int>> training_set = training_data();
    int set_num = 1;

    // Build vocabulary
    for (auto &data : training_set) {
        std::vector<std::string> words = preprocessing(data.first);
        for (auto &word : words) {
            vocabulary.insert(word);
        }
    }
    
    // Convert set to vector for indexing
    vocablist.assign(vocabulary.begin(), vocabulary.end());
    
    // Create feature vectors
    for (auto &data : training_set) {
        std::vector<std::string> words = preprocessing(data.first);
        std::vector<int> temp_vec(vocablist.size(), 0);
        
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

void LogisticReg::model_trainer() {
    weights.assign(vocablist.size(), 0.0);
    b = 0.1;
    
    for (int epoch = 0; epoch < 1000; epoch++) {
        double total_loss = 0.0;
        
        for (auto &sentence : wordsvec) {
            std::vector<int> temp = sentence.second.first;
            int label = sentence.second.second;
            double wx = 0.0;

            for (int i = 0; i < weights.size(); i++) {
                wx += weights[i] * temp[i];
            }
            
            double z = wx + b;
            // Prevent numerical overflow
            if (z > 30.0) z = 30.0;
            if (z < -30.0) z = -30.0;
            
            double y = 1.0 / (1.0 + std::exp(-z));
            
            // Update weights and bias
            for (int i = 0; i < weights.size(); i++) {
                weights[i] += lr * (label - y) * temp[i];
            }
            b += lr * (label - y);
            
            // Calculate loss for monitoring
            double loss = -(label * std::log(y + 1e-10) + (1 - label) * std::log(1 - y + 1e-10));
            total_loss += loss;
        }
        
        // Optional: Print loss every 100 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / wordsvec.size() << std::endl;
        }
    }
}

std::string LogisticReg::predictor(const std::string& input) {
    std::vector<std::string> words = preprocessing(input);
    std::vector<int> x(vocablist.size(), 0);
    
    for (auto &word : words) {
        auto it = std::find(vocablist.begin(), vocablist.end(), word);
        if (it != vocablist.end()) {
            int index = std::distance(vocablist.begin(), it);
            x[index] = 1;
        }
    }

    double wx = 0.0;
    for (int i = 0; i < weights.size(); i++) {
        wx += x[i] * weights[i];
    }
    
    double z = wx + b;
    double y = 1.0 / (1.0 + std::exp(-z));

    if (y > 0.5) {
        return "positive";
    } else {
        return "negative";
    }
}

void LogisticReg::call_logisticreg() {
    std::string input;
    logistic_reg();
    model_trainer();
    
    std::cout << "Model training completed. Vocabulary size: " << vocablist.size() << std::endl;
    
    while (true) {
        std::cout << "Enter a sentence (type 'exit' to quit): ";
        std::getline(std::cin, input);
        if (input == "exit") break;
        
        std::string prediction = predictor(input);
        std::cout << "Prediction: " << prediction << std::endl;
    }
}