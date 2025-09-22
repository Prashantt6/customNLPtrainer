#include "spamclassifier.h"
#include<vector>
#include<string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <unordered_set>
#include <iterator>

// Training data call function
std::vector<std::pair<std::string, int>> SpamClassifier::training_data() {
    std::vector<std::pair<std::string, int>> data = {
        {"Win a free iPhone now!!!", 1},
        {"You have been selected for a lottery", 1},
        {"Call this number to claim your prize", 1},
        {"Hey, are we still meeting tomorrow?", 0},
        {"Don't forget the assignment submission", 0},
        {"Let's have lunch together", 0}
    };

    return data;
}


// Tokenizes the sentences into words
std::vector<std::string> SpamClassifier::tokenizer(const std::string& sentence) {
    std::vector<std::string> tokens;
    std::istringstream iss(sentence);
    std::string temp;

    // Splitting sentence by spaces and converting to lowercase
    while (iss >> temp) {
        for (char &c : temp) { c = tolower(c); }
        tokens.push_back(temp);
    }

    return tokens;
}

// All the stopwords are removed and tokenization is done
std::vector<std::string> SpamClassifier::preprocessing(const std::string& sentence) {
    std::vector<std::string> words = tokenizer(sentence);
    std::vector<std::string> result;

    // Only keep words that are not in stopwords
    for (const auto& word : words) {
        if (stopwords.find(word) == stopwords.end()) {
            result.push_back(word);
        }
    }
    return result;
}

// Creating a Bag of Words model for training dataset
void SpamClassifier::BOW() {
    std::vector<std::pair<std::string, int>> training_set = training_data();

    for (auto &data : training_set) {
        std::vector<std::string> words = preprocessing(data.first);

        if (data.second == 1) { // If sentence is spam
            spamdocs++;
            for (auto& word : words) {
                spamWordCount[word]++;   // Count frequency of word in spam messages
                totalspamWords++;        // Count total words in spam messages
            }
        } 
        else { // If sentence is not spam
            notspamdocs++;
            for (auto& word : words) {
                notspamWordCount[word]++;  // Count frequency of word in non-spam messages
                totalnotspamWords++;       // Count total words in non-spam messages
            }
        }
    }
}


// Prediction for user input sentence using Naive Bayes
std::string SpamClassifier::predictor(const std::string &input) {
    std::vector<std::string> words = preprocessing(input);
    double spamscore = 0, notspamscore = 0;

    // Calculate probability for each word in input sentence
    for (auto &word : words) {
        // Laplace smoothing applied to avoid zero probabilities
        double spamProb = (spamWordCount[word] + 1.0) / (totalspamWords + spamWordCount.size());
        double notspamProb = (notspamWordCount[word] + 1.0) / (totalnotspamWords + notspamWordCount.size());

        spamscore += log(spamProb);        // Add log probability for spam
        notspamscore += log(notspamProb);  // Add log probability for not spam
    }

    // Add prior probabilities (based on document counts)
    int totalDocs = spamdocs + notspamdocs;
    spamscore += log(((double)spamdocs) / totalDocs);
    notspamscore += log(((double)notspamdocs) / totalDocs);

    // Final classification decision
    if (spamscore > notspamscore) return "spam";
    else return "not spam";
}


// Wrapper function to run classifier interactively
void SpamClassifier::Classifier_call() {
    BOW();  // Build the Bag of Words model from training data
    std::string input;

    // Take user input repeatedly until "exit" is typed
    while (true) {
        std::cout << "Enter the sentence (type exit to quit) :  ";
        std::getline(std::cin, input);

        if (input == "exit") exit(0);  // Exit condition

        std::string Prediction = predictor(input);
        std::cout << "The sentence is : " << Prediction << std::endl;
    }
}
