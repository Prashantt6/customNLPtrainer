

#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>


class SpamClassifier{
    private :
        const std::unordered_set<std::string> stopwords = {
        "is", "to", "a", "be", "am", "i", "are", "my", "in", "the", "and", "or"
    };
    std::vector<std::string> tokenizer (const std::string& sentence);  
    std::vector<std::pair<std::string, int>> training_data();
    std::vector<std::string> preprocessing (const std::string & sentence);
    std::unordered_map<std::string, int> spamWordCount;
    std::unordered_map<std::string, int> notspamWordCount;
    
    int totalspamWords = 0;
    int totalnotspamWords = 0;
    int spamdocs = 0;
    int notspamdocs = 0;


    public:
        void BOW();
        std::string predictor(const std::string &input);
        void Classifier_call();

};