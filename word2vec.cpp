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
#include <random>
#include <cmath>

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

void word2vec :: makepair(const std::vector<std::string>& training_set){
    
    for(auto& data : training_set){
        std::vector<std::string> words = preprocessing(data);
        for( int i = 0 ; i< words.size() ; i++ ){
            std::string target = words[i];
            int left = std::max(0 , i - window);
            int right = std::min((int)words.size() - 1 , i + window);
            for( int j = left ; j <= right ; j++){
                if (i == j) continue;
                training_pairs.push_back({target , words[j]});
            }
         }
    }
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
            if(wordsvec.find(word) == wordsvec.end()){
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
    makepair(training_set);

}

std::vector<std::vector<float>> initialize_matrix( int rows , int columns) {
    std::vector<std::vector<float>> mat(rows , std::vector<float>(columns));
    float limit = 1.0 / std::sqrt(columns);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit , limit);

    for(int i = 0 ; i< rows ; i++){
        for( int j = 0 ;  j < columns ; j++){
            mat[i][j] = dist(gen);
        }
    }
    return mat;
}

int word_id(std::vector<std::string>& vocablist ,std::string& target){
    for(int i = 0 ; i< vocablist.size() ; i++){
        if(vocablist[i] == target){
            return i;
        }
        
    }
    return -1;
}

std::vector<float> multiply (const std::vector<float>& h , const std::vector<std::vector<float>>& W2 ){
    int V = W2[0].size();
    int D = h.size();
    std::vector<float> u(V , 0.0);
    for(int i = 0 ; i< V ; i++){
        for( int j = 0 ; j< D  ; j++){
            u[i] +=   h[j] * W2[j][i]; //taking W2 as a flat vector
        }
    }
    return u;
}


void word2vec :: prediction(){
    int V = vocablist.size();
    int D = 50;
    auto W1 = initialize_matrix(V , D);
    auto W2 = initialize_matrix(D , V );
    std::vector<float>expo;
    std::vector<float>prob;
    
    
    // Forward propagation using SKIP_Gram method   
    for(int epoch = 0 ; epoch <1000 ; epoch++){
        for(auto& word : training_pairs){
            float total_loss = 0.0;
            std::string target = word.first;
            std::string context = word.second;
            int wordindex = word_id(vocablist , target);
            auto h = W1[wordindex];
            auto u = multiply(h , W2 );
            expo.clear();
            float sum = 0;
            for(int i = 0 ; i < V ; i++){
                float ex = exp(u[i]);
                expo.push_back(ex);
                sum += ex;
            }
            prob.clear();
            for(int i = 0 ; i < V ; i++){
                float p = expo[i]/ sum;
                prob.push_back(p);
            }

            // Calculating loss 
            int context_index = word_id(vocablist , context);
            float P = expo[context_index] /sum;
            float loss = -log(P);
            total_loss += loss ;
            
            // Loss for each epoch
            std::cout << "Epoch " << epoch  << " - Avg Loss: " << total_loss / training_pairs.size()  << std::endl;

            
        }
    }   

} 

