#include <string>
#include <vector>
#include <unordered_map>

// Naive Bayes classifier for sentiment analysis (Positive, Negative, Neutral)
class NaiveBayes {
private:
    // Word frequency maps for each class
    std::unordered_map<std::string, int> positiveWordCount;
    std::unordered_map<std::string, int> negativeWordCount;
    std::unordered_map<std::string, int> neutralWordCount;

    // Total word counts per class (used in probability calculation)
    int totalPositiveWords = 0;
    int totalNegativeWords = 0;
    int totalNeutralWords = 0;

    // Provides the training dataset: sentence + label (1=positive, -1=negative, 0=neutral)
    std::vector<std::pair<std::string , int >> training_data();

    // Breaks a sentence into lowercase words (tokenizer)
    std::vector<std::string> tokenizer(std::string sentence);

    // Removes stopwords and prepares a clean list of words
    std::vector<std::string> preprocessing(const std::string &sentence);

public:
    // Train the Naive Bayes model using the training data
    void trainer(std::vector<std::pair<std::string, int>>& trainingData);

    // Predict the sentiment of a given input sentence
    std::string predictor(std::string& sentence);

    // Wrapper to train the model and run an interactive prediction loop
    void Bayescall();

    // Optional: calculate probabilities for words (not implemented in .cpp yet)
    void probability_calculator();
};
