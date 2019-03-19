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
#ifndef CS_DATA_REPORTER_HDF5_H
#define CS_DATA_REPORTER_HDF5_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "../cs_model.h"
#include <hdf5.h>

namespace cs
{
class csDataReporter_hdf5 : public csDataReporter_t
{
public:
    csDataReporter_hdf5(const std::string& hdf5FileName_res,
                        const std::string& hdf5FileName_der);
    virtual ~csDataReporter_hdf5();

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
    hid_t                    fileResults;
    hid_t                    fileDerivatives;
    hid_t                    datatypeFloat;
    std::string              m_hdf5FileNameResults;
    std::string              m_hdf5FileNameDervatives;
    real_t                   m_currentTime;
    std::vector<real_t>      m_times;
};
}

#endif
