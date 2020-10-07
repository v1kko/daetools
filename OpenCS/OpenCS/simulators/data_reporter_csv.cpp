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
#include "data_reporter_csv.h"
#include "cs_data_reporters.h"
#include <fstream>
#include <iomanip>

namespace cs
{
std::shared_ptr<csDataReporter_t> createDataReporter_CSV(const std::string& csvFileName_res,
                                                         const std::string& csvFileName_der,
                                                         char delimiter,
                                                         const std::string& format,
                                                         int precision)
{
    return std::shared_ptr<csDataReporter_t>(new csDataReporter_csv(csvFileName_res,
                                                                    csvFileName_der,
                                                                    delimiter,
                                                                    format,
                                                                    precision));
}

csDataReporter_csv::csDataReporter_csv(const std::string& csvFileName_res,
                                       const std::string& csvFileName_der,
                                       char delimiter,
                                       const std::string& format,
                                       int precision)
{
    m_currentTime         = 0.0;
    reportResults         = false;
    reportDerivatives     = false;
    m_fileNameResults     = csvFileName_res;
    m_fileNameDerivatives = csvFileName_der;

    m_delimiter = delimiter;
    m_format    = format;
    m_precision = precision;
}

csDataReporter_csv::~csDataReporter_csv()
{
    Disconnect();
}

std::string	csDataReporter_csv::GetName() const
{
    return "CSV";
}

bool csDataReporter_csv::Connect(int rank)
{
    pe_rank = rank;

    reportResults = false;
    reportDerivatives = false;

    fileResults.open(m_fileNameResults.c_str());
    if(fileResults.is_open())
    {
        reportResults = true;
        if(m_format == "fixed")
            fileResults << std::setiosflags(std::ios_base::fixed);
        else if(m_format == "scientific")
            fileResults << std::setiosflags(std::ios_base::scientific);
        fileResults << std::setprecision(m_precision);
    }

    fileDerivatives.open(m_fileNameDerivatives.c_str());
    if(fileDerivatives.is_open())
    {
        reportDerivatives = true;
        if(m_format == "fixed")
            fileDerivatives << std::setiosflags(std::ios_base::fixed);
        else if(m_format == "scientific")
            fileDerivatives << std::setiosflags(std::ios_base::scientific);
        fileDerivatives << std::setprecision(m_precision);
    }

    return true;
}

bool csDataReporter_csv::IsConnected()
{
    return true;
}

bool csDataReporter_csv::Disconnect()
{
    reportResults     = false;
    reportDerivatives = false;
    if(fileResults.is_open())
    {
        fileResults.flush();
        fileResults.close();
    }
    if(fileDerivatives.is_open())
    {
        fileDerivatives.flush();
        fileDerivatives.close();
    }
    return true;
}

bool csDataReporter_csv::RegisterVariables(const std::vector<std::string>& variableNames)
{
    int Nequations = variableNames.size();
    fileResults << " ";
    for(int i = 0; i < Nequations; i++)
        fileResults << m_delimiter << i;
    fileResults << std::endl;

    fileResults << "time";
    for(int i = 0; i < Nequations; i++)
        fileResults << m_delimiter << variableNames[i];
    fileResults << std::endl;

    fileDerivatives << " ";
    for(int i = 0; i < Nequations; i++)
        fileDerivatives << m_delimiter << i;
    fileDerivatives << std::endl;

    fileDerivatives << "time";
    for(int i = 0; i < Nequations; i++)
        fileDerivatives << m_delimiter << "d" << variableNames[i] << "/dt";
    fileDerivatives << std::endl;

    return true;
}

bool csDataReporter_csv::StartNewResultSet(real_t time)
{
    m_currentTime = time;
    return true;
}

bool csDataReporter_csv::EndOfData()
{
    return true;
}

bool csDataReporter_csv::SendVariables(const real_t* values, const size_t Nequations)
{
    static int counter = 0;

    if(!reportResults)
        return true;

    fileResults << m_currentTime;
    for(int i = 0; i < Nequations; i++)
        fileResults << m_delimiter << values[i];
    fileResults << std::endl;

    fileResults.flush();

    counter++;

    return true;
}

bool csDataReporter_csv::SendDerivatives(const real_t* derivatives, const size_t Nequations)
{
    static int counter_dx = 0;

    if(!reportDerivatives)
        return true;

    fileDerivatives << m_currentTime;
    for(int i = 0; i < Nequations; i++)
        fileDerivatives << m_delimiter << derivatives[i];
    fileDerivatives << std::endl;

    fileDerivatives.flush();

    counter_dx++;

    return true;
}

}
