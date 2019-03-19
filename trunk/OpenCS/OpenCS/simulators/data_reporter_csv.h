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
#ifndef CS_DATA_REPORTER_CSV_H
#define CS_DATA_REPORTER_CSV_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "../cs_model.h"

namespace cs
{
class csDataReporter_csv : public csDataReporter_t
{
public:
    csDataReporter_csv(const std::string& csvFileName_res,
                       const std::string& csvFileName_der,
                       char delimiter            = ';',
                       const std::string& format = "fixed",
                       int precision             = 15);
    virtual ~csDataReporter_csv();

public:
    std::string	GetName() const;
    bool Connect(int rank);
    bool IsConnected();
    bool Disconnect();
    bool RegisterVariables(const std::vector<std::string>& variableNames);
    bool StartNewResultSet(real_t time);
    bool EndOfData();
    bool SendVariables(const real_t* values, const size_t n);
    bool SendDerivatives(const real_t* derivatives, const size_t n);

protected:
    int                      pe_rank;
    bool                     reportResults;
    bool                     reportDerivatives;
    std::ofstream            fileResults;
    std::ofstream            fileDerivatives;
    std::string              m_fileNameResults;
    std::string              m_fileNameDerivatives;
    real_t                   m_currentTime;

    char        m_delimiter;
    std::string m_format;
    int         m_precision;
};

}

#endif
