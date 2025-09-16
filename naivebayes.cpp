#include "naivebayes.h"
#include<vector>
#include<string>
#include<sstream>
#include<iostream>
#include<algorithm>
#include<unordered_map>
#include<cmath>



std::vector<std::pair<std::string , int >> NaiveBayes::training_data(){
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
    "is" , "to" , "a" , "be"
};
std::vector<std::string>NaiveBayes::tokenizer(std::string sentence ){
    std::istringstream iss (sentence);
    std::string temp;
    std::vector<std::string> tokens;
    while(iss >> temp){
        for(char &c :temp) {c = tolower(c);}
        tokens.push_back(temp);
    }
    return tokens;
}

std::vector<std::string> NaiveBayes::preprocessing(const std::string& sentence) {
    std::vector<std::string> words = tokenizer(sentence);
    std::vector<std::string> result;
    
    for (const auto& word : words) {
        if (std::find(stopwords.begin(), stopwords.end(), word) == stopwords.end()) {
            result.push_back(word);
        }
    }
    return result;
}
void NaiveBayes:: trainer(std::vector<std::pair< std::string , int >> &training_data){
    totalPositiveWords = 0;
    totalNegativeWords = 0;
    totalNeutralWords = 0;
   
    for( auto &data : training_data){
        std::vector<std::string>words = preprocessing(data.first);
        if(data.second == 1){
           for(auto &word : words){
            positiveWordCount[word]++;
            totalPositiveWords++;
           }

        }
        else if(data.second == -1){
           for(auto &word : words){
            negativeWordCount[word]++;
            totalNegativeWords++;
           }

        }
        else if(data.second == 0){
           for(auto &word : words){
            neutralWordCount[word]++;
            totalNeutralWords++;
           }

        }   

    }
    
   std::cout << "Training completed. Total words: " 
              << totalPositiveWords + totalNegativeWords + totalNeutralWords 
              << std::endl;
}
std::string NaiveBayes::predictor(std::string& input){
    std::vector<std::string>words =preprocessing(input);
    double posScore =0 , negScore = 0 , neuScore = 0;
    for (auto &word : words){
        double wordPosProb = (positiveWordCount[word]+1.0)/(totalPositiveWords + positiveWordCount.size());
        double wordNegProb = (negativeWordCount[word]+1.0 )/ (totalNegativeWords + negativeWordCount.size());
        double wordNeuProb = (neutralWordCount[word]+1.0)/ (totalNeutralWords + neutralWordCount.size ());

        posScore += log(wordPosProb);
        negScore += log(wordNegProb);
        neuScore += log(wordNeuProb);

    }
    double totalWords = totalPositiveWords + totalNegativeWords + totalNeutralWords;
    posScore += log(((double)totalPositiveWords + 1.0 )/(totalWords + 3.0));
    negScore += log(((double)totalNegativeWords + 1.0) / (totalWords + 3.0));
    neuScore += log(((double)totalNeutralWords + 1.0) /( totalWords + 3.0));


    if(posScore> negScore && posScore > neuScore){
        return "Positive";
    }
    else if(  negScore > posScore && negScore > neuScore){
        return "Negative";
    }
    else {
        return "Neutral";
    }
}

void NaiveBayes:: Bayescall(){
    std::vector<std::pair<std::string , int >> training_set = training_data();
    this-> trainer(training_set);
    std::string input;
    while(true){
    std::cout<<"Enter the sentence(type exit to quit ) : ";
    std::getline(std::cin , input);
    if(input == "exit") exit(0);
    std::string Prediction = predictor(input);
    std::cout<<std::endl;
    std::cout<<"Sentiment of : "<< input << " is " <<" '"<< Prediction << "'"<<std::endl;
    }
 }
