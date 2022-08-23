//
// Created by Bismarck on 2021/10/18 0018.
//

#ifndef RC2021_TCPCLIENT_H
#define RC2021_TCPCLIENT_H

#include <string>


enum tcpDataType {
    teamId=0,
    shibie=1,
    celiang=2
};


class tcpClient {
public:
    tcpClient(const std::string& addr, unsigned short port);
    ~tcpClient();
    void send(tcpDataType dataType, const std::string& info);

private:
    int sock=0;
    bool connected;
    bool connect(const std::string& addr, unsigned short port);
};


#endif //RC2021_TCPCLIENT_H
