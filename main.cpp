#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
#include<map>
#include<unordered_map>
#include "naivebayes.h"
#include "logisticregression.h"
#include "spamclassifier.h"
#include "word2vec.h"

// Global vectors and maps for word embeddings / mapping
std::vector <std::string> words;                   // Stores tokenized words
std::vector<int> numseq;                           // Stores numerical representation of words
std::unordered_map<std::string , int > word2id;    // Maps words to unique IDs
std::unordered_map<int , std::string> id2word;     // Maps IDs back to words

// Initialize vocabulary with predefined words and assign IDs
void vocabulary(){
    word2id["i"] = 1;
    word2id["love"] = 2;
    word2id["nepal"] = 3;
    word2id["you"] = 4;
    word2id["hate"] = 5;
    word2id["cpp"] = 6;

    // Build reverse mapping for convenience
    for(const auto& pair : word2id){
        id2word[pair.second] = pair.first;
    }
}

// Converts a word to its ID, returns -1 if word not found
int word_to_id(const std::string &word ){
    auto it = word2id.find(word);
    if(it != word2id.end()){
        return it->second;
    }
    else {
        return -1;
    }
}

// Converts ID back to word, returns "UNK" if ID not found
std::string id_to_word(int num){
    auto it = id2word.find(num);
    if(it != id2word.end()){
        return it->second;
    }
    else {
        return "UNK";
    }
}

// Tokenizes a sentence into lowercase words
std::vector<std::string> tokenize(const std::string &input){
    std::vector<std::string> words;
    std::istringstream iss(input);
    std::string temp;
    
    while(iss >> temp){
        for(char &c : temp ){
            c = tolower(c);       // Convert each character to lowercase
        }
        words.push_back(temp);
    }
    return words;
}

// Converts tokenized words to their numeric IDs
std::vector<int> words_to_num(const std::vector<std::string> &words){
    std::vector<int> numseq;
    for(const auto& word : words){
        int num = word_to_id(word);   // Get ID for each word
        numseq.push_back(num);
    }
    return numseq;
}

// Displays words from their numeric representation
void display(const std::vector<int> &numseq){
    for(int i : numseq){
        std::string word = id_to_word(i);
        std::cout<<word<<" ";
    }
    std::cout<<std::endl;
}

// Prints the total word count of a sentence
void wordcount(const std::vector<std::string> words){
    std::cout<<"Word_Count :" << words.size()<<std::endl;
}

// Provides default responses for certain keywords
void defrespond(const std::vector<int> numseq){
    for(int i : numseq){
        if(i == 2){                  // Word "love"
            std::cout<<"love is a beautiful thing"<<std::endl;
        }
        else if( i ==3 ){             // Word "nepal"
            std::cout<<"Nepal is a amazing country"<<std::endl;
        }
    }
}

// Creates a Bag-of-Words representation and calculates simple sentiment
void bagofwords(const std::vector<int> numseq){
    int vocab = 6;
    std::vector<int> bow(vocab , 0);

    // Count occurrences of each word
    for(int i : numseq){
        if(i>0){
            bow[i-1]++;
        }
    }

    // Predefined sentiment values for words
    int sentiment = 0;
    std::unordered_map<int , int> word_sentiment;
    word_sentiment[1] = 0;
    word_sentiment[2] = 1;
    word_sentiment[3] = 0;
    word_sentiment[4] = 0;
    word_sentiment[5] = -1;
    word_sentiment[6] = 0;

    // Multiply word count by sentiment and sum up
    for(int i : numseq ){
        if(word_sentiment.find(i) != word_sentiment.end()){
            sentiment += bow[i-1] * word_sentiment[i];
        }
    }

    // Output overall sentiment
    if(sentiment >0){
        std::cout<<"Positive statement"<<std::endl;
    }
    else if(sentiment < 0){
        std::cout<<"Negative statement"<<std::endl;
    }
    else {
        std::cout<<"Neutral statement"<<std::endl;
    }
}

// Runs all word-based operations: count, display, default response, Bag-of-Words
void model(const std::vector<std::string> words, const std::vector<int>numseq){
    wordcount(words);          // Print word count
    display(numseq);           // Display words
    defrespond(numseq);        // Default responses
    bagofwords(numseq);        // Sentiment via Bag-of-Words
}

// Global word counts for simple sentiment predictor
std::unordered_map<std::string , int > postive_count;
std::unordered_map<std::string , int > negative_count;
std::unordered_map<std::string , int > neutral_count;

// Predict sentiment based on word counts from training
std::string sentiment_predictor(std::vector<std::string> words){
    int pos_score = 0 , neg_score = 0 , neu_score = 0;

    // Sum up scores based on trained word counts
    for(auto &word : words){
        if(postive_count.find(word) != postive_count.end()){
            pos_score += postive_count[word];
        }
        if(negative_count.find(word) != negative_count.end()){
            neg_score += negative_count[word];
        }
        if(neutral_count.find(word) != neutral_count.end()){
            neu_score += neutral_count[word];
        }
    }

    // Return the class with highest score
    if(pos_score > neg_score && pos_score > neu_score ) return "Positive";
    else if(neg_score > pos_score && neg_score > neu_score) return "Negative";
    else return "Neutral";
}

int  main(){
    // -------------------------
    // Uncomment any section below to run specific models
    // -------------------------

    // Word-to-number model
    // std::string input;
    // std::cout<<":";
    // std::getline(std::cin , input);
    // vocabulary();
    // std::vector<std::string> words= tokenize(input);
    // std::vector<int> numseq = words_to_num(words);
    // model(words, numseq);

    // Naive Bayes model
    // NaiveBayes nb;
    // nb.Bayescall();

    // Logistic Regression model
    // LogisticReg lr ;
    // lr.call_logisticreg();

    // Spam classifier model
    // SpamClassifier sc;
    // sc.Classifier_call();    // Start interactive spam detection loop


    // Word to vector 

    word2vec wv;
    wv.word2vec_call();
    return 0;
}
