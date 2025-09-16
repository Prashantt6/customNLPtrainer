#include <string>
#include <vector>
#include <unordered_map>

class NaiveBayes {
private:
    std::unordered_map<std::string, int> positiveWordCount;
    std::unordered_map<std::string, int> negativeWordCount;
    std::unordered_map<std::string, int> neutralWordCount;
    
    
    

    int totalPositiveWords = 0;
    int totalNegativeWords = 0;
    int totalNeutralWords = 0;
    std::vector<std::pair<std::string , int >> training_data();
    std::vector<std::string> tokenizer(std::string sentence);
    std::vector<std::string>preprocessing(const std::string &sentence);


public:
    void trainer(std::vector<std::pair<std::string, int>>& trainingData);
    std::string predictor( std::string& sentence);
    void Bayescall();
    void probability_calculator();
};
