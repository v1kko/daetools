/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_LOGS_H
#define CS_LOGS_H

#include <string>
#include <iostream>
#include <fstream>
#include "../cs_model.h"

namespace cs
{
class csLog_StdOut : public csLog_t
{
public:
    csLog_StdOut();
    virtual ~csLog_StdOut();

public:
    bool Connect(int rank);
    void Disconnect();
    bool IsConnected();
    std::string	GetName(void) const;
    void Message(const std::string& strMessage);

protected:
    int         pe_rank;
    std::string m_strName;
};

class csLog_TextFile : public csLog_t
{
public:
    csLog_TextFile(const std::string& strFileName);
    virtual ~csLog_TextFile();

public:
    bool Connect(int rank);
    void Disconnect();
    bool IsConnected();
    std::string GetName(void) const;
    void Message(const std::string& strMessage);

    std::string GetFilename() const;

protected:
    int           pe_rank;
    std::string   m_strName;
    std::ofstream file;
    std::string   m_strFilename;
};

}

#endif
