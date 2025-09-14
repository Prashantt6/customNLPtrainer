#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
#include<map>
std::vector <std::string> words;
std::vector<int> numseq;

int vocabulary(std::string word){
    std::map<std::string, int>  word2id;
    word2id["I"] = 1;
    word2id["love"] = 2;
    word2id["Nepal"] = 3;
    word2id["You"] = 4;

     

    auto it = word2id.find(word);
    if( it != word2id.end()){
        return it->second;
    }
    else {
        return -1;
    }

}

void tokenize(std::string input){
    
    std::istringstream iss(input);  
    std::string temp;
    while (iss >> temp) {
        words.push_back(temp);
    }

}
void numsequence(){
    while(!words.empty()){
        std::string word = words.front();
        words.erase(words.begin());
        int num = vocabulary(word);
        numseq.push_back(num);
    }
    while(!numseq.empty()){
        int last = numseq.front();
        numseq.erase(numseq.begin());
        std::cout<<last << " ";
    }
    
}
void forwardpass();

void backpass();

int  main(){
    std::string input;
    std::cout<<":";
    std::getline(std::cin , input);
    tokenize(input);
    numsequence();
    
}
