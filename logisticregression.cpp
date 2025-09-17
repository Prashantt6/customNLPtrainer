#include "logisticregression.h"
#include<vector>
#include<string>
#include<sstream>
#include<iostream>
#include<algorithm>
#include<unordered_map>
#include<cmath>
#include<unordered_set>



std::vector<std::pair<std::string , int >>LogisticReg:: training_data(){
    std::vector<std::pair<std::string , int>> data{
        {"I love nepal ", 1},
        {"I hate cpp", -1},
        {"I live in nepal", 0},
        {"I love my country", 1},
        {"cpp is good", 1},
        {"cpp is a programming language ", 0},
        {"I hate eating vegetables", -1},
        {"cpp is sometimes boring", -1},
        {"I like you", 1},
        {"I dislike you", -1},
        {"I am not interested in talking", -1},
        {" I like nepal", 1}

    };
    return data;
}   


std::vector<std::string> stopwords ={
    "is" , "to" , "a" , "be","am" , "i" , "are" , "my" , "in"
};
std::vector<std::string>LogisticReg:: tokenizer(std::string sentence ){
    std::istringstream iss (sentence);
    std::string temp;
    std::vector<std::string> tokens;
    while(iss >> temp){
        for(char &c :temp) {c = tolower(c);}
        tokens.push_back(temp);
    }
    return tokens;
}

std::vector<std::string>LogisticReg:: preprocessing(const std::string& sentence) {
    std::vector<std::string> words = tokenizer(sentence);
    std::vector<std::string> result;
    
    for (const auto& word : words) {
        if (std::find(stopwords.begin(), stopwords.end(), word) == stopwords.end()) {
            result.push_back(word);
        }
    }
    return result;
}

void LogisticReg::logistic_reg(){
    std::vector<std::pair<std::string , int>> training_set = training_data();
    std::vector<std::string > BOW_vec;
    int set_num= 1;

     for(auto &data : training_set){
        std::vector<std::string> words = preprocessing(data.first);
        for(auto &word : words){
            vocabulary.insert(word);
        }
    }


    for(auto &data : training_set){

        std::vector<std::string> words = preprocessing(data.first);
        std::vector<std::string>vocablist(vocabulary.begin(), vocabulary.end());
        std::vector<int> temp_vec(vocablist.size(), 0);
        for(auto &word : words){
            auto it = std::find(vocablist.begin() , vocablist.end (), word);
            if(it != vocablist.end()){
                int index = it - vocablist.begin();
                temp_vec[index] = 1;
            }
        }
        wordsvec["set" + std::to_string(set_num)] = temp_vec;
        set_num++;

        
}