#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>

// Simple Naive Bayes Spam Classifier
class SpamClassifier {
    private:
        // List of stopwords to ignore while processing text
        const std::unordered_set<std::string> stopwords = {
            "is", "to", "a", "be", "am", "i", "are", "my", "in", "the", "and", "or"
        };

        // Splits a sentence into lowercase words
        std::vector<std::string> tokenizer(const std::string& sentence);

        // Returns training dataset (sentence + label)
        // label = 1 → spam, 0 → not spam
        std::vector<std::pair<std::string, int>> training_data();

        // Removes stopwords and returns cleaned words
        std::vector<std::string> preprocessing(const std::string& sentence);

        // Word counts for spam and non-spam sentences
        std::unordered_map<std::string, int> spamWordCount;
        std::unordered_map<std::string, int> notspamWordCount;

        // Total word counts for spam and non-spam categories
        int totalspamWords = 0;
        int totalnotspamWords = 0;

        // Number of spam and non-spam documents
        int spamdocs = 0;
        int notspamdocs = 0;


    public:
        // Build bag-of-words model from training data
        void BOW();

        // Predicts if input text is spam or not spam
        std::string predictor(const std::string &input);

        // Interactive function that keeps asking user for input
        void Classifier_call();
};
