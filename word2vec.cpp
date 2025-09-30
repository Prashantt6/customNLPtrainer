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
#include <map>

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
                std::vector<float>temp_vec(vocablist.size() , 0);
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
    int D = W2.size();
    std::vector<float> u(V , 0.0);
    for(int i = 0 ; i< V ; i++){
        for( int j = 0 ; j< D  ; j++){
            u[i] +=   h[j] * W2[j][i]; //taking W2 as a flat vector
        }
    }
    return u;
}

std::vector<std::vector<float>>  word2vec :: forward_pass(int V , int D , std::vector<std::vector<float>>& W1 , std::vector<std::vector<float>>& W2){

        for(int epoch = 0 ; epoch <1000 ; epoch++){
            total_loss = 0.0;
            for(auto& word : training_pairs){
                
                std::string target = word.first;
                std::string context = word.second;
                int wordindex = word_id(vocablist , target);
                auto h = W1[wordindex];
                auto u = multiply(h , W2 );
                expo.clear();
                float sum = 0;
                float max_u = *std::max_element(u.begin() , u.end());
                for(int i = 0 ; i < V ; i++){
                    float ex = exp(u[i] - max_u);
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
                float loss = -log(prob[context_index] + 1e-10);
                total_loss += loss ;
                
                // Backward pass 
                backward_pass(h , W1 , W2 , target , context);
                
                
            }
            // Loss for each epoch
            // std::cout << "Epoch " << epoch  << " - Avg Loss: " << total_loss / training_pairs.size()  << std::endl;  
        } 

        return W1;
        

    
}

void word2vec :: backward_pass(std::vector<float>& h ,std::vector<std::vector<float>>& W1 , std::vector<std::vector<float>>& W2, std::string& target , std::string& context){
    int D = h.size();
    int V = prob.size();
    std::vector<float>error(V , 0.0   ) ;
    std::vector<float> tempvec = wordsvec[context];
    for(int i = 0 ; i < prob.size() ; i++){
        error[i] = prob[i] - tempvec[i];
    }
    

    // Gradient for W2
    std::vector<std::vector<float>>del_W2(D , std::vector<float>(V , 0.0));
    for( int i = 0 ; i < D ; i++){
        for(int j = 0 ; j < V ; j++){
            del_W2[i][j] = h[i] * error[j];
            W2[i][j] -= lr * del_W2[i][j];
        }
    }

    // Gradient for W1
    std::vector<float>del_h(D , 0.0);

    for(int i = 0 ; i< D ; i++){
        for(int j = 0 ; j< V ; j++){
            del_h[i] += error[j] * W2[i][j];

        }
       
    }
    for (auto &val : del_h) {
        if (val > 5.0) val = 5.0;
        if (val < -5.0) val = -5.0;
    }

    int target_index = word_id(vocablist , target);
    if(target_index != -1 ){
        for(int i = 0 ; i < D ; i++){
            W1[target_index][i] -= lr * del_h[i];
        }
    }



}


void word2vec :: prediction( ){
    int V = vocablist.size();
    int D = 20;
    auto W1 = initialize_matrix(V , D);
    auto W2 = initialize_matrix(D , V );
    
    // Forward propagation using SKIP_Gram method   
   std::vector<std::vector<float>> Word2vec =  forward_pass(V , D , W1 , W2);

    for(int j = 0 ; j <Word2vec.size() ; j++){
        wordsvec[vocablist[j]] = Word2vec[j];
    }
    

} 

void word2vec :: vecofword(){
    std::string input;
    std::cout << "Enter a word : ";
    std::getline(std::cin, input);
    for(char& c : input) { c = std::tolower(c); }
    
    display(input);
}
float dotproduct(std::vector<float>& A , std::vector<float>& B){
    float result = 0 ;
    for(int i = 0 ; i < A.size() ; i++){
        result += A[i] * B[i];
    }
    return result;
}
float Magnitude(std::vector<float>& vec){
    float  M = 0;
    for(float x : vec){
        M += x * x ;
    }
    return std::sqrt(M); 
}

void word2vec :: most_similar(){
    std::map<std::string , float> similarity_map;
    std::string input;
    std::cout<< "Enter the word : ";
    std::getline(std::cin , input);
    for(char&c : input){ c = std::tolower(c);}
    
    std::vector<float> inputvec;
    if (wordsvec.find(input) != wordsvec.end()) {
        inputvec = wordsvec[input];
    } else {
        std::cout << "Input word is not in vocabulary" << std::endl;
        return;
    }
    
    for(auto& word : wordsvec){
        if( word.first != input){
            float dot_prod = dotproduct(word.second , inputvec);
            float magnitude_of_word = Magnitude(word.second);
            float magnitude_of_input = Magnitude(inputvec);
            // std::cout<< " Dot product of " << word.first <<" "<<  input << " is " << dot_prod <<std::endl;
            // std::cout<< " Magnitude of vector of " << word.first << " is " << magnitude_of_word <<std::endl;
            // std::cout<< " Magnitude of vector of "<< input << " is " << magnitude_of_input << std::endl;
            float Cosine_similarity = dot_prod / (magnitude_of_input * magnitude_of_word);
            similarity_map[word.first ] = Cosine_similarity;
        } 
        
    }

    // Sorting similarity map in ascending order 
    std::vector<std::pair<std::string , float >>vec (similarity_map.begin() , similarity_map.end());

    std::sort(vec.begin() , vec.end(),[](const auto& a, const auto& b ){
        return a.second > b.second ;
    });
    
    std::cout<<" The top similar words with " << input << " is : " <<std::endl;
    int limit = std::min(5, (int)vec.size());
    for( int i = 0 ; i < limit ; i++){
        std::cout<<vec[i].first << " : " <<vec[i].second << std::endl;
    }
    

    

}
void word2vec:: display(std::string& word){
        if (wordsvec.find(word) != wordsvec.end()) {
            for (auto &num : wordsvec[word]) std::cout << num << " ";
            std::cout << std::endl;
         } 
         else {
        std::cout << "Not in vocabulary\n";
         }


}



void word2vec :: word2vec_call(){
    training();
    prediction();
    // std::string input;
    // while (true) {
    //     std::cout << "Enter a word (type 'exit' to quit): ";
    //     std::getline(std::cin, input);
    //     for(char& c : input) { c = std::tolower(c); }
    //     if (input == "exit") break;
       
    //     prediction(input);
        
    // }
    int ch ;
    do{
        std::cout<<" 1. Vector of a word "<<std::endl
                <<" 2. Most Similar  "<<std::endl 
                <<" 3. Exit ";
        std::cout<<std::endl;
        std::cout<<"Enter the choice : ";
        std::cin >> ch;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        switch(ch){
            case 1 :
                vecofword();
                break;
            case 2 :
                most_similar();
                break;
            case 3 :
                break;
            default :
                std::cout<<"Invalid choice !!"<<std::endl;
                break;


        }      
    }while(ch != 3);
   
    


}

