#include "spamclassifier.h"
#include<vector>
#include<string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <unordered_set>
#include <iterator>

std::vector<std::pair<std::string, int>> SpamClassifier::training_data() {
    std::vector<std::pair<std::string, int>> data = {
        {"Win a free iPhone now!!!", 1},
        {"You have been selected for a lottery", 1},
        {"Call this number to claim your prize", 1},
        {"Hey, are we still meeting tomorrow?", 0},
        {"Don't forget the assignment submission", 0},
        {"Let's have lunch together", 0}
    };

    return data;
}

std::vector<std::string>SpamClassifier::tokenizer(const std::string& sentence){
    std::vector<std::string> tokens;
    std::istringstream iss (sentence);
    std::string temp ;
    while(iss >> temp){
        for(char &c : temp ) { c = tolower(c);}
        tokens.push_back(temp);

    }
    return tokens;
}
std::vector<std::string> SpamClassifier :: preprocessing(const std::string& sentence){
    std::vector<std::string> words = tokenizer(sentence);
    std::vector<std::string> result;
    for(const auto& word : words){
        if(stopwords.find(word) == stopwords.end()){
            result.push_back(word);
        }

    }
    return result;
}
void SpamClassifier :: BOW(){
    std::vector<std::pair<std::string, int>> training_set = training_data();

    for(auto &data : training_set)
    {
        std::vector<std::string> words = preprocessing(data.first);
        if(data.second == 1) {
           for(auto& word : words ){
                positiveWordCount[word]++;
                totalPositiveWords++;
           }
        }
        else {
            for(auto& word : words){
                negativeWordCount[word]++;
                totalNegativeWords++;
            }
        }
    }
    
    
}