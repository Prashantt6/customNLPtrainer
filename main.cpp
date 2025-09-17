#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
#include<map>
#include<unordered_map>
#include "naivebayes.h"
#include "logisticregression.h"
std::vector <std::string> words;
std::vector<int> numseq;
std::unordered_map<std::string , int > word2id;
std::unordered_map<int , std::string> id2word;
void vocabulary(){
    
    word2id["i"] = 1;
    word2id["love"] = 2;
    word2id["nepal"] = 3;
    word2id["you"] = 4;
    word2id["hate"] = 5;
    word2id["cpp"] = 6;

    for(const auto& pair : word2id){
        id2word[pair.second] = pair.first;
    }

}
// std::unordered_map<int , int > sentiment_value;
// void wordsvocab(){
//     sentiment_value[1] = 0;
//     sentiment_value[2] = 1;
//     sentiment_value[3] = 0;
//     sentiment_value[4] = 0;
//     sentiment_value[5] = -1;
//     sentiment_value[6] = 0;

// }
// int id_to_value(int num){
//     auto it = sentiment_value.find(num);
//     if(it != sentiment_value.end()){
//         return it->second;
//     }
//     else {
//         return 0;
//     }
// }

int word_to_id(const std::string &word ){
    auto it = word2id.find(word);
    if(it != word2id.end()){
        return it->second;
    }
    else {
        return -1;
    }
}

std::string id_to_word(int num){
    auto it = id2word.find(num);
    if(it != id2word.end()){
        return it->second;

    }
    else {
        return "UNK";
    }
}
std::vector<std::string> tokenize(const std::string &input){
    std::vector<std::string> words;
    
    std::istringstream iss(input);
    std::string temp;
    
    while(iss >> temp){
        for(char &c : temp ){
        c = tolower(c);
        }
        words.push_back(temp);
    }
    return words;

}

std::vector<int> words_to_num(const std::vector<std::string> &words){
    std::vector<int> numseq;
    for(const auto& word : words){
        int num = word_to_id(word);
        numseq.push_back(num);
    }
   
    return numseq;
    
}
void display(const std::vector<int> &numseq){
    for(int i : numseq){
        std::string word = id_to_word(i);
        std::cout<<word<<" ";
    }
    std::cout<<std::endl;
}
void wordcount(const std::vector<std::string>words){
    std::cout<<"Word_Count :" << words.size()<<std::endl;
}

void defrespond(const std::vector<int> numseq){
    for(int i : numseq){
        if(i == 2){
            std::cout<<"love is a beautiful thing"<<std::endl;
        }
        else if( i ==3 ){
            std::cout<<"Nepal is a amazing country"<<std::endl;
        }
    }
}
void bagofwords(const std::vector<int> numseq){
    
    int vocab = 6;
    std::vector<int>bow(vocab , 0);
    for(int i : numseq){
        if(i>0){
            bow[i-1]++;
        }
    }
    int sentiment = 0;
    std::unordered_map<int , int> word_sentiment;
    word_sentiment[1] = 0;
    word_sentiment[2] = 1;
    word_sentiment[3] = 0;
    word_sentiment[4] = 0;
    word_sentiment[5] = -1;
    word_sentiment[6] = 0;

    for(int i : numseq ){
        if(word_sentiment.find(i) != word_sentiment.end()){
            sentiment += bow[i-1] * word_sentiment[i];
        }
    }
    
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

void model(const std::vector<std::string> words, const std::vector<int>numseq){
    wordcount(words);
    display(numseq);
    defrespond(numseq);
    bagofwords(numseq);
    
    
}
// std::vector<std::pair<std::string , int >> training_data(){
//     std::vector<std::pair<std::string , int>> data{
//         {"I love nepal ", 1},
//         {"I hate cpp", -1},
//         {"I live in nepal", 0},
//         {"I love my country", 1},
//         {"cpp is good", 1},
//         {"cpp is a programming language ", 0},
//         {"I hate eating vegetables", -1},
//         {"cpp is sometimes boring", -1},
//         {"I like you", 1},
//         {"I dislike you", -1},
//         {"I am not interested in talking", -1},
//         {" I like nepal", 1}

//     };
//     return data;
// }
std::unordered_map<std::string , int > postive_count;
std::unordered_map<std::string , int > negative_count;
std::unordered_map<std::string , int > neutral_count;

// std::string tokenizer(std::string words){
//     std::istringstream iss (words);
//     std::string temp;
//     while(iss >> temp){
//         for(char &c : temp ){c = tolower(c);}
//         return temp;
//     }
// }

// void trainer(std::vector<std::pair<std::string , int >> training_set){
    
//     for(auto &data : training_set){
        
//         if(data.second == 1){
            
//             std::istringstream iss (data.first);
//             std::string temp;
            
//             while(iss >> temp){
//                 for(char &c : temp){
//                 c = tolower(c);
//                 }
//                 postive_count[temp]++;
//             }
                        
//         }
//         else if( data.second == -1){
//             std::istringstream iss (data.first);
//             std::string temp;
//             while(iss >> temp){
//                 for(char &c : temp){
//                 c = tolower(c);
//                 }
//                 negative_count[temp]++;
//             }
//         }
//         else {
//             std::istringstream iss (data.first);
//             std::string temp;
//             while(iss >> temp){
//                 for(char &c : temp){
//                 c = tolower(c);
//                 }
//                 neutral_count[temp]++;
//             }
//         }
//     }
    
// }
std::string sentiment_predictor(std::vector<std::string>words){
    int pos_score = 0 , neg_score = 0 , neu_score = 0;
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
    if(pos_score > neg_score && pos_score > neu_score ) return "Positive";
    else if(neg_score > pos_score && neg_score > neu_score) return "Negative";
    else return "Neutral";


}

int  main(){
    // std::string input;
    // std::cout<<":";
    // std::getline(std::cin , input);
    // vocabulary();
    // std::vector<std::string> words= tokenize(input);
    // std::vector<int> numseq = words_to_num(words);

    // model(words, numseq);

    // Training data
    // std::vector<std::pair<std::string , int >> training_set = training_data();
    // trainer(training_set);

    
    // Asking user to input 
    // std::string input ;
    // while(true){
    //     std::cout<<"Enter the sentence(type exit to quit) : ";
    //     std::getline(std::cin , input);
    //     if(input == "exit") exit(0);
    //     std::vector<std::string> words = tokenize(input);
    //     std::string Sentiment = sentiment_predictor(words);
    //     std::cout<<std::endl;
    //     std::cout<<"Input : "<<input<<std::endl<<"Prediction : "<<Sentiment<<std::endl;
    

    // }

    // Calling for naive bayes
    NaiveBayes nb;
    nb.Bayescall();
    
    return 0;
    


    
}
