#include <cstdio>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/algorithm/string_regex.hpp>
#include <boost/lexical_cast.hpp>
#include "include/encode.hpp"
#include "fmi_component.h"
using boost::asio::ip::tcp;

/***************************************************
       daeFMIComponent_t
***************************************************/
daeFMIComponent_t::daeFMIComponent_t()
{
    debugMode = false;
    m_ptcpipSocket.reset(new tcp::socket(m_ioService));
}

daeFMIComponent_t::~daeFMIComponent_t()
{
    if(m_ptcpipSocket)
        m_ptcpipSocket->close();
}

void daeFMIComponent_t::logFatal(const std::string& error)
{
    printf(error.c_str());

    if(functions && loggingOn && functions->logger)
        functions->logger(functions->componentEnvironment, instanceName.c_str(), fmi2Fatal, "logStatusFatal", error.c_str());
}

void daeFMIComponent_t::logError(const std::string& error)
{
    printf(error.c_str());

    if(functions && loggingOn && functions->logger)
        functions->logger(functions->componentEnvironment, instanceName.c_str(), fmi2Error, "logStatusError", error.c_str());
}

void daeFMIComponent_t::executeQuery(const std::string& queryParameters,
                                     std::string& queryResponseHeaders,
                                     std::string& queryResponseContent)
{
    if(!m_ptcpipSocket)
        throw std::runtime_error("TCP/IP socket object is null");

    tcp::resolver resolver(m_ioService);
    tcp::resolver::query query(webServiceAddress, boost::lexical_cast<std::string>(webServicePort));
    boost::system::error_code ec = boost::asio::error::host_not_found;

    // Make 'noRetries' attempts to connect.
    // Wait for 'retryAfter' milli-seconds before the next attempt.
    int noRetries  = 10;
    int retryAfter = 1000; // ms

    for(int i = 0; i < noRetries; i++)
    {
        tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
        tcp::resolver::iterator end;
        ec = boost::asio::error::host_not_found;
        while(ec && endpoint_iterator != end)
        {
            m_ptcpipSocket->close();
            m_ptcpipSocket->connect(*endpoint_iterator++, ec);
        }

        // Break the for loop if the connection has been established.
        if(!ec)
            break;

        // Wait for 'retryAfter' before the next attempt.
        boost::this_thread::sleep(boost::posix_time::milliseconds(retryAfter));
    }
    if(ec)
        throw std::runtime_error((boost::format("Cannot establish a connection to the server at %s/%s:%d") % webServiceAddress % webServiceName % webServicePort).str());

    boost::asio::streambuf request;
    std::ostream request_stream(&request);

    std::string html_content = queryParameters;
    request_stream << (boost::format("POST /%s HTTP/1.1\r\n") % webServiceName).str();
    request_stream << (boost::format("Host: %s %d\r\n") % webServiceAddress % webServicePort).str();
    request_stream <<                "Accept: application/json\r\n";
    request_stream << (boost::format("Content-Length: %d\r\n") % html_content.length()).str();
    request_stream <<                "Content-Type: application/x-www-form-urlencoded\r\n";
    request_stream << "\r\n";
    request_stream << html_content;

    // Send the request.
    boost::asio::write(*m_ptcpipSocket, request, ec);
    if(ec)
        throw boost::system::system_error(ec);

    // Read the whole response
    boost::asio::streambuf response;
    size_t nReceived = boost::asio::read(*m_ptcpipSocket, response, ec);
    if(ec.value() != boost::asio::error::eof)
        throw boost::system::system_error(ec);

    response.commit(nReceived);

    std::istream is(&response);
    char* buffer = new char[nReceived+1];
    buffer[nReceived] = '\0';
    is.read(buffer, nReceived);
    std::string str_response = buffer;
    delete[] buffer;

    // Split the received http headers and the content
    std::vector<std::string> arr_header_contents;
    boost::algorithm::split_regex(arr_header_contents, str_response, boost::regex("\r\n\r\n"));
    if(arr_header_contents.size() < 2)
        throw std::runtime_error((boost::format("Invalid response received:\n%s") % str_response).str());

    queryResponseHeaders = arr_header_contents[0];
    queryResponseContent = arr_header_contents[1];

/*
    if(line.find_first_of("HTTP") != std::string::npos)
    {
        if(line.find("200 OK") != std::string::npos)
            httpStatusCode = 200;
        else if(line.find("500 Internal Server Error") != std::string::npos)
            httpStatusCode = 500;
        queryResponseHeader.push_back(line);
    }
    else if(line.find_first_of("Date") != std::string::npos)
        queryResponseHeader.push_back(line);
    else if(line.find_first_of("Server") != std::string::npos)
        queryResponseHeader.push_back(line);
    else if(line.find_first_of("Content-type") != std::string::npos)
        queryResponseHeader.push_back(line);
    else if(line.find_first_of("Content-Length") != std::string::npos)
        queryResponseHeader.push_back(line);
*/
}
