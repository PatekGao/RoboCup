//
// Created by Bismarck on 2021/10/18 0018.
//

#include "scheduler/tcpClient.h"
#include <unistd.h>
#include <arpa/inet.h>
#include <iostream>
#include <cstring>

using std::string;
using std::cout;
using std::endl;


tcpClient::tcpClient(const std::string& addr, unsigned short port) {
    cout << "Connecting to " << addr << ":" << port << endl;
    connected = connect(addr, port);
    while (!connected) {
        sleep(1);
        cout << "Retry connect to " << addr << ":" << port << endl;
        connected = connect(addr, port);
    }
}

tcpClient::~tcpClient() {
    close(sock);
}

bool tcpClient::connect(const std::string& addr, unsigned short port) {
    sockaddr_in serv_addr{};

    sock = socket(PF_INET, SOCK_STREAM, 0);
    if(sock == -1){
        return false;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(addr.c_str());
    serv_addr.sin_port = htons(port);

    if(::connect(sock, (struct sockaddr*) &serv_addr, sizeof(serv_addr))==-1) {
        cout << "Couldn't connect to " << addr << ":" << port << endl;
        return false;
    } else {
        cout << "Successfully connected to " << addr << ":" << port << endl;
        return true;
    }
}

void tcpClient::send(tcpDataType dataType, const std::string &info) {
    cout<<"info: "<<info<<endl;
    int len = (int)info.length();
    cout<<"info length: "<<info.length()<<endl;
    char *data = new char[len + 9];
    *(unsigned int*)data = htonl((unsigned int)dataType);
    cout<<"datatype: "<<(unsigned int)dataType<<endl;
    *(unsigned int*)(data+4) = htonl((unsigned int)len);
    memcpy(data+8, info.c_str(), len);
    data[len + 8] = 0;

    write(sock, data, len+8);

    delete[] data;
}
