#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>

class LogisticReg {
private:
    double b = 0.1;
    double lr = 0.01;
    std::set<std::string> vocabulary;
    std::unordered_map<std::string, std::pair<std::vector<int>, int>> wordsvec;
    std::vector<std::string> vocablist;
    std::vector<double> weights;

    const std::unordered_set<std::string> stopwords = {
        "is", "to", "a", "be", "am", "i", "are", "my", "in"
    };

    std::vector<std::pair<std::string, int>> training_data();
    std::vector<std::string> tokenizer(const std::string& sentence);
    std::vector<std::string> preprocessing(const std::string& sentence);

public:
    void logistic_reg();
    void model_trainer();
    void call_logisticreg();
    std::string predictor(const std::string& input);
};
