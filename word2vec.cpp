#include "word2vec.h"
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <unordered_set>
#include <iterator>
#include <fstream>

// Loading training dataset through trainingdata.txt
std::vector<std::string>word2vec :: load_training_data(const std::string &trainingdata) {
    std::ifstream data(trainingdata);
    std::vector<std::string> training_set;
    std::string line ;

    if(!data.is_open()){
        std::cerr<<"Training data not loaded" <<std::endl;
        return training_set;
    }

    while(std::getline(data , line)){
        if(!line.empty()){
            training_set.push_back(line);

        }
    }
    data.close();
    return training_set;
}

std::vector<std::string> word2vec :: tokenizer (const std::string &sentence){
    std::istringstream iss(sentence);
    std::string temp;
    std::vector<std::string> tokens;

    while (iss >> temp) {
        for (char& c : temp) { c = std::tolower(c); } // convert to lowercase
        if (!temp.empty()) {
            tokens.push_back(temp);
        }
    }
    return tokens;
}

std::vector<std::string> word2vec :: preprocessing (const std::string &sentence)
{
    std::vector<std::string> words = tokenizer(sentence);
    std::vector<std::string> result;

    for (const auto& word : words) {
        if (stopwords.find(word) == stopwords.end()) {
            result.push_back(word);
        }
    }
    return result;
}

void word2vec :: training(){
    std::vector<std::string> training_set = load_training_data("trainingdata.txt");

    // Step 1: Build vocabulary from all training sentences
    for (auto &data : training_set) {
        std::vector<std::string> words = preprocessing(data);
        for (auto &word : words) {
            vocabulary.insert(word);
        }
    }

    // Step 2: Convert vocabulary (set) to a list for indexing
    vocablist.assign(vocabulary.begin(), vocabulary.end());

    for(auto& data : training_set){
        std::vector<std::string> words = preprocessing(data);
        
        for(auto &word : words ){
            std::vector<int>temp_vec(vocablist.size() , 0);
            auto it = std::find(vocablist.begin() , vocablist.end() , word );
            if( it != vocablist.end()){
                int index = std::distance(vocablist.begin() , it);
                temp_vec[index] = 1;
            }
            wordsvec[word] = temp_vec;
        }
    }

}
