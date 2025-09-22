#include "naivebayes.h"
#include<vector>
#include<string>
#include<sstream>
#include<iostream>
#include<algorithm>
#include<unordered_map>
#include<cmath>

// Training dataset: pairs of sentence + label (1 = positive, -1 = negative, 0 = neutral)
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

// Stopwords that should be ignored (not meaningful for classification)
std::vector<std::string> stopwords ={
    "is" , "to" , "a" , "be"
};

// Break a sentence into lowercase words (basic tokenizer)
std::vector<std::string>NaiveBayes::tokenizer(std::string sentence ){
    std::istringstream iss (sentence);
    std::string temp;
    std::vector<std::string> tokens;
    while(iss >> temp){
        for(char &c :temp) {c = tolower(c);} // lowercase conversion
        tokens.push_back(temp);
    }
    return tokens;
}

// Preprocessing: tokenize + remove stopwords
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

// Training function: counts word frequencies for positive, negative, and neutral classes
void NaiveBayes:: trainer(std::vector<std::pair< std::string , int >> &training_data){
    totalPositiveWords = 0;
    totalNegativeWords = 0;
    totalNeutralWords = 0;
   
    for( auto &data : training_data){
        std::vector<std::string>words = preprocessing(data.first);
        if(data.second == 1){ // positive sentence
           for(auto &word : words){
            positiveWordCount[word]++;
            totalPositiveWords++;
           }
        }
        else if(data.second == -1){ // negative sentence
           for(auto &word : words){
            negativeWordCount[word]++;
            totalNegativeWords++;
           }
        }
        else if(data.second == 0){ // neutral sentence
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

// Prediction function: applies Naive Bayes formula to classify input
std::string NaiveBayes::predictor(std::string& input){
    std::vector<std::string>words = preprocessing(input);
    double posScore =0 , negScore = 0 , neuScore = 0;

    // Calculate probabilities for each word
    for (auto &word : words){
        double wordPosProb = (positiveWordCount[word]+1.0)/(totalPositiveWords + positiveWordCount.size());
        double wordNegProb = (negativeWordCount[word]+1.0 )/ (totalNegativeWords + negativeWordCount.size());
        double wordNeuProb = (neutralWordCount[word]+1.0)/ (totalNeutralWords + neutralWordCount.size ());

        // use log to avoid underflow in multiplication
        posScore += log(wordPosProb);
        negScore += log(wordNegProb);
        neuScore += log(wordNeuProb);
    }

    // Add class priors
    double totalWords = totalPositiveWords + totalNegativeWords + totalNeutralWords;
    posScore += log(((double)totalPositiveWords + 1.0 )/(totalWords + 3.0));
    negScore += log(((double)totalNegativeWords + 1.0) / (totalWords + 3.0));
    neuScore += log(((double)totalNeutralWords + 1.0) /( totalWords + 3.0));

    // Pick class with highest probability
    if(posScore > negScore && posScore > neuScore){
        return "Positive";
    }
    else if(negScore > posScore && negScore > neuScore){
        return "Negative";
    }
    else {
        return "Neutral";
    }
}

// Wrapper: trains model then runs interactive prediction loop
void NaiveBayes:: Bayescall(){
    std::vector<std::pair<std::string , int >> training_set = training_data();
    this-> trainer(training_set);
    std::string input;
    while(true){
        std::cout<<"Enter the sentence(type exit to quit ) : ";
        std::getline(std::cin , input);
        if(input == "exit") exit(0); // exits on "exit"
        std::string Prediction = predictor(input);
        std::cout<<std::endl;
        std::cout<<"Sentiment of : "<< input << " is " <<" '"<< Prediction << "'"<<std::endl;
    }
}
