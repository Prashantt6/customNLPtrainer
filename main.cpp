#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
#include<map>
#include<unordered_map>
std::vector <std::string> words;
std::vector<int> numseq;
std::unordered_map<std::string , int > word2id;
std::unordered_map<int , std::string> id2word;
void vocabulary(){
    
    word2id["I"] = 1;
    word2id["love"] = 2;
    word2id["Nepal"] = 3;
    word2id["You"] = 4;

    for(const auto& pair : word2id){
        id2word[pair.second] = pair.first;
    }

}
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
}
void wordcount(const std::vector<std::string>words){
    std::cout<<"Word_Count :" << words.size()<<std::endl;
}

void model(const std::vector<std::string> words, const std::vector<int>numseq){
    wordcount(words);
    display(numseq);
    
    
}



int  main(){
    std::string input;
    std::cout<<":";
    std::getline(std::cin , input);
    vocabulary();
    std::vector<std::string> words= tokenize(input);
    std::vector<int> numseq = words_to_num(words);

    model(words, numseq);


    
}
