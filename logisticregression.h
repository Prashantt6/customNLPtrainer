#include<string>
#include<vector>
#include<set>

class LogisticReg {
    private :
         std::vector<std::pair<std::string , int >> training_data();
         std::vector<std::string> tokenizer(std::string sentence);
         std::vector<std::string>preprocessing(const std::string &sentence);
         std::set<std::string> vocabulary;
         std::unordered_map<std::string, std::pair<std::vector<int>, int>> wordsvec;
         std::vector<std::string>vocablist;
    public :
        void logistic_reg();
        void model_trainer();



};