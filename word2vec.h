#include <vector>
#include <unordered_set>
#include <string>
#include <fstream>
#include <unordered_map>



class word2vec {
    private :
        std::unordered_set<std::string> vocabulary;
        std::vector<std::string> vocablist;

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

        std::vector<std::string> load_training_data(const std::string &training_data);

        int window = 5;

        std::vector<std::pair<std::string , std::string >> training_pairs;


        std::unordered_map<std::string, std::vector<int>> wordsvec;

    public: 
        void training();
        void makepair(const std::vector<std::string>& training_set);
        

};