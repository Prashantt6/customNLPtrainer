

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
    std::unordered_map<std::string, int> positiveWordCount;
    std::unordered_map<std::string, int> negativeWordCount;
    
    int totalPositiveWords = 0;
    int totalNegativeWords = 0;


    public:
        void BOW();
};