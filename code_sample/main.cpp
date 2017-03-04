//
//  Created by Vladimir P on 11/8/16.
//  Copyright © 2016 Vladimir P. All rights reserved.
//

#include <string>
#include <regex>
#include <iostream>
#include <utility>
#include <assert.h>
#include <queue>
#include <map>

using namespace std; // bad style but improves readability in this toy problem

// tag describes one complete line of an html file
class tag{
    bool opening; // is this an opening tag or a closing
    string name;
    map<string,string> attributes;
public:
    tag(const string&);
    bool is_opening() const { return opening;}
    string get_name() const { return name;}
    map<string,string> get_attributes() const { return attributes;}
};

// str - a complete line of input showing a tag
// constructor tag() parses string into class tag
tag::tag(const string& str){
    regex open_name("^<([^ <>/]+)"); // <tag-name value1 = "1" value2 = "2">
    regex attr("([^ <>/]+) = \"([^ <>/]+)\"");
    regex clos_name("^</([^ <>/]+)>"); // </tag-name>
    
    smatch sm;
    if (regex_search(str, sm, open_name)){
        name = sm[1];
        opening = true;
        
        auto attr_begin = sregex_iterator(str.begin(), str.end(), attr);
        for (auto it = attr_begin; it != sregex_iterator(); ++it) {
            sm = *it;
            attributes[sm[1]] = sm[2];
        }
    } else{
        assert(regex_search(str, sm, clos_name));
        name = sm[1];
        opening = false;
    }
}

// Element describes everything in an html file between
// the element's opening tag and the matching closing tag.
// For example, the whole html file is an element.
class element{
    tag root_tag; // the opening tag
    string name; // the opening tag's name
    map<string, element> children;
// Tags between the opening tag and the closing
// constitue a collection of elements themself.
// children - is the collection of these elements accessed by their names
public:
    element(deque<tag>);
    map<string,string> get_attributes() const {
        return root_tag.get_attributes();
    }
    map<string, element> get_children() const { return children;}
};

// html - is a deque of tag starting from the opening tag and
// ending with the closing tag
element::element(deque<tag> html): root_tag(html[0]),
name(root_tag.get_name()){
    html.pop_back();
    html.pop_front();
    
    deque<tag> subhtml;
    int depth = 0;
    while(html.size()){
        subhtml.push_back(html.front());
        depth += html.front().is_opening() ? 1 : -1;
        assert(depth >= 0);
        html.pop_front();
        if(depth == 0){
            children[subhtml.front().get_name()] = element(subhtml);
            subhtml.clear();
        }
    }
}

namespace raw_input{ // the very first stage of parsing
    
    typedef queue<string> request;
    
    // parses a request into a queue of names
    // e.g. tag1.tag2~name into {tag1, tag2, name}
    request request_parser(string str){
        request res;
        if(str.find('~') == string::npos) return res;
        
        regex delimiter("[.~]"); // tag1.tag2~name
        auto attr_begin = sregex_token_iterator(str.begin(), str.end(),
                                                delimiter, -1);
        for (auto it = attr_begin; it != sregex_token_iterator(); ++it) {
            res.push(*it);
        }
        return res;
    }
    
    // parses whole input into
    // html as element and
    // requests as queue< request >
    pair<element, queue<request> > parser(){
        deque<tag> html;
        queue<request> requests;
        int html_len, requests_len;
        
        cin >>  html_len >> requests_len;
        cin.ignore(256, '\n');
        
        string temp;
        for(int i=0; i< html_len; i++){
            assert(getline(cin, temp));
            html.push_back(tag(temp));
        }
        html.push_front(tag("<my_html>"));
        html.push_back(tag("</my_html>"));
        for(int i=0; i< requests_len; i++){
            assert(getline(cin, temp));
            requests.push(request_parser(temp));
        }
        
        return make_pair(element(html), requests);
    }
    
} // namespace raw_input

// output - the desired string to print
// Processes a request given some html and produces the output to print.
string request_processing(const element& html, raw_input::request& request){
    if(!request.size()) return "Not Found!";
    
    string front = request.front();
    request.pop();
    
    if(request.size()){
        if(!html.get_children().count(front))
            return "Not Found!";
        else
            return request_processing(html.get_children()[front], request);
    } else{
        if(!html.get_attributes().count(front))
            return "Not Found!";
        else
            return html.get_attributes()[front];
    }
}

int main()
{
    auto input = raw_input::parser();
    const element html = input.first;
    queue<raw_input::request> requests = input.second;

    while(!requests.empty()){
        cout << request_processing(html, requests.front()) << endl;
        requests.pop();
    }
    return 0;
}
