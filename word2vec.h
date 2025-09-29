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

        

        // Splits a sentence into lowercase words
        std::vector<std::string> tokenizer(const std::string& sentence);

        // Removes stopwords from tokenized sentence
        std::vector<std::string> preprocessing(const std::string& sentence);

        std::vector<std::string> load_training_data(const std::string &training_data);

        int window = 5;

        int embedding_size = 10;

        double lr = 0.001;

        float total_loss;

        std::vector<float>expo;

        std::vector<float>prob;

        std::vector<std::pair<std::string , std::string >> training_pairs;


        std::unordered_map<std::string, std::vector<float>> wordsvec;

    public: 
        void training();
        void makepair(const std::vector<std::string>& training_set);
        std::vector<std::vector<float>> forward_pass(int V , int D ,std::vector<std::vector<float>>& W1 , std::vector<std::vector<float>>& W2 );
        
        void backward_pass(std::vector<float>& h ,std::vector<std::vector<float>>& W1, std::vector<std::vector<float>>& W2 , std::string& target , std::string& context );
        void prediction();
        void display(std::string& word);
        void vecofword();
        void most_similar();
        
        void word2vec_call();

};