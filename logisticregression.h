#include<string>
#include<vector>
#include<set>

class LogisticReg {
    private :
         std::vector<std::pair<std::string , int >> training_data();
         std::vector<std::string> tokenizer(std::string sentence);
         std::vector<std::string>preprocessing(const std::string &sentence);
         std::set<std::string> vocabulary;
         std::unordered_map<std::string , std::vector<int>> wordsvec;
    public :
        void logistic_reg();



};